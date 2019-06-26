#!/usr/bin/env python
from __future__ import print_function
import sys
import tf
import mil_ros_tools
import cv2
import numpy as np
import rospy
from sub8_perception.cfg import ServantIdentifierConfig
from std_msgs.msg import Header
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridgeError
from sub8_vision_tools import MultiObservation
from image_geometry import PinholeCameraModel
from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import PoseStamped, Pose, Point
from sub8_msgs.srv import VisionRequest, VisionRequestResponse
from mil_ros_tools import Image_Subscriber, Image_Publisher
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server as DynamicReconfigureServer

'''
This Vampiric Grymoire identifies the Servants of the Enemy:

    Bats  --> Found at the bottom of a PVC container, we drop garlic on it
    Wolves   --> Found at the bottom of a PVC container, we drop garlic on it

One of the sevants will be hidden behind a lever. To get max points,
all garlic must be dropped into this hidden servant.

General outline of task
    Find exposed enemy. 
    If confident, pull lever Kronk.
    Find enemy that was not exposed.
    Center on enemy.
    Bombs away!
'''

class ServantIdent:

    def __init__(self):

        # Pull constants from config file
        self.dyn_lower = [0, 0, 0]
        self.dyn_upper = [0, 0, 0]
        self.max_contour_area = 0
        self.min_contour_area = 0
        self.camera = rospy.get_param('~camera_topic',
                                      'camera/front/left/image_rect_color')
        self.goal = None
        self.last_config = None
        self.reconfigure_server = DynamicReconfigureServer(ServantIdentifierConfig, self.reconfigure)

        # Image Subscriber and Camera Information

        self.image_sub = Image_Subscriber(self.camera, self.image_cb)
        self.camera_info = self.image_sub.wait_for_camera_info()

        self.camera_info = self.image_sub.wait_for_camera_info()
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)
        self.frame_id = self.camera_model.tfFrame()

        # Ros Services so mission can be toggled and info requested
        rospy.Service('~enable', SetBool, self.toggle_search)
        self.image_pub = Image_Publisher("vamp_vision/debug")
        self.mask_image_pub = rospy.Publisher(
            'vamp_vision/mask', Image, queue_size=1)

        # Debug
        self.debug = rospy.get_param('~debug', True)


    @staticmethod
    def parse_string(threshes):
        ret = [float(thresh.strip()) for thresh in threshes.split(',')]
        if len(ret) != 3:
            raise ValueError('not 3')
        return ret

    def reconfigure(self, config, level):
            try:
                self.override = config['override']
                self.lower = self.parse_string(config['dyn_lower'])
                self.upper = self.parse_string(config['dyn_upper'])
                self.min_contour_area = config['min_cont']
                self.max_contour_area = config['max_cont']

            except ValueError as e:
                rospy.logwarn('Invalid dynamic reconfigure: {}'.format(e))
                return self.last_config

            if self.override:
                # Dynamic Values for testing
                self.lower = np.array(self.lower)
                self.upper = np.array(self.upper)
            else:
                self.lower = rospy.get_param('~servant_low_thresh', [100, 100, 80])
                self.upper = rospy.get_param('~servant_high_thresh', [255, 255, 80])
                
            self.last_config = config
            rospy.loginfo('Params succesfully updated via dynamic reconfigure')
            return config

    def image_cb(self, image):
        '''
        Run each time an image comes in from ROS.
        '''
        if not self.enabled:
            return

        self.last_image = image

        if self.last_image_time is not None and \
                self.image_sub.last_image_time < self.last_image_time:
            # Clear tf buffer if time went backwards (nice for playing bags in
            # loop)
            self.tf_listener.clear()

        self.last_image_time = self.image_sub.last_image_time
        self.acquire_targets(image)

    def toggle_search(self, srv):
        '''
        Callback for standard ~enable service. If true, start
        looking at frames for buoys.
        '''
        if srv.data:
            rospy.loginfo("TARGET ACQUISITION: enabled")
            self.enabled = True

        else:
            rospy.loginfo("TARGET ACQUISITION: disabled")
            self.enabled = False

        return SetBoolResponse(success=True)


    def detect(self, c):
        '''
        Verify the shape in the masked image is large enough to be a valid target.
        This changes depending on target Vampire, as does the number of targets we want.  
        '''
        target = "unidentified"
        peri = cv2.arcLength(c, True)

        if peri < self.min_contour_area or peri > self.max_contour_area:
            return target
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        target = "Target Aquisition Successful"

        return target

    def mask_image(self, cv_image, lower, upper):
        mask = cv2.inRange(cv_image, lower, upper)
        # Remove anything not within the bounds of our mask
        output = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        if (self.debug):
            try:
                # print(output)
                self.mask_image_pub.publish(
                    self.bridge.cv2_to_imgmsg(np.array(output), 'bgr8'))
            except CvBridgeError as e:
                print(e)

        return output

    def acquire_targets(self, cv_image):
        # Take in the data and get its dimensions.
        height, width, channels = cv_image.shape

        # create NumPy arrays from the boundaries
        lower = np.array(self.lower, dtype="uint8")
        upper = np.array(self.upper, dtype="uint8")

        # Generate a mask based on the constants.
        blurred = self.mask_image(cv_image, lower, upper)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # Compute contours
        cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[1]
        '''
        We use OpenCV to compute our contours and then begin processing them
        to ensure we are identifying a proper target.
        '''

        shape = ''
        peri_max = 0
        max_x = 0
        max_y = 0
        m_shape = ''
        for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = .000001
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
            shape = self.detect(c)

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image

            c = c.astype("float")
            # c *= ratio
            c = c.astype("int")
            if shape == "Target Aquisition Successful":
                if self.debug:
                    try:
                        cv2.drawContours(cv_image, [c], -1, (0, 255, 0), 2)
                        cv2.putText(cv_image, shape, (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 2)
                        self.image_pub.publish(cv_image)
                    except CvBridgeError as e:
                        print(e)


def main(args):
    rospy.init_node('servant_ident', anonymous=False)
    ServantIdent()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
