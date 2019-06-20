#!/usr/bin/env python
from __future__ import print_function
import sys
import tf
import mil_ros_tools
import cv2
import numpy as np
import rospy
from sub8_perception.cfg import DraculaIdentifierConfig
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
This Vampiric Grymoire identifies Dracula himself:
    Dracula's Heart --> Heart Shaped hole where we fire a "stake"
    Dracula's Head --> A orange lever that we slide to decapitate the beast

General outline of task
    Find torpedo board with Hydrophones.
    Find Heart.
    Align torpedo with heart.
    Fire!
    If confident, zoom out.
    Find pole.
    Figure out what side of Dracula pole is on.
    Go to it and push it to the opposite side. 
'''

class DracIdent:

    def __init__(self):

        # Pull constants from config file
        self.dyn_lower = [0, 0, 0]
        self.dyn_upper = [0, 0, 0]
        self.min_trans = 0
        self.max_velocity = 0
        self.timeout = 0
        self.min_observations = 0
        self.camera = rospy.get_param('~camera_topic',
                                      'CLAHE/debug')
        self.goal = None
        self.last_config = None
        self.reconfigure_server = DynamicReconfigureServer(DraculaIdentifierConfig, self.reconfigure)

    @staticmethod
    def parse_string(threshes):
        ret = [float(thresh.strip()) for thresh in threshes.split(',')]
        if len(ret) != 3:
            raise ValueError('not 3')
        return ret

    def reconfigure(self, config, level):
        try:
            self.override = config['override']
            self.goal = config['target']
            self.heart_lower = self.parse_string(config['heart_lower'])
            self.heart_upper = self.parse_string(config['heart_upper'])
            self.handle_lower = self.parse_string(config['handle_lower'])
            self.handle_upper = self.parse_string(config['handle_upper'])
            self.min_trans = config['min_trans']
            self.max_velocity = config['max_velocity'] 
            self.timeout = config['timeout']
            self.min_observations = config['min_obs']

        except ValueError as e:
            rospy.logwarn('Invalid dynamic reconfigure: {}'.format(e))
            return self.last_config

        if self.override
            # Dynamic Values for testing
            self.lower = np.array(dyn_lower)
            self.upper = np.array(dyn_upper)
        else:
            # Hard Set for use in Competition
            self.heart_lower = rospy.get_param('~heart_low_thresh', [0, 0, 80])
            self.heart_upper = rospy.get_param('~heart_high_thresh', [0, 0, 80])
            self.handle_lower = rospy.get_param('~handle_low_thresh', [0, 0, 80])
            self.handle_upper = rospy.get_param('~handle_high_thresh', [0, 0, 80])
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

    def request_buoy(self, srv):
        '''
        Callback for 3D vision request. Uses recent observations of target
        board  specified in target_name to attempt a least-squares position
        estimate. Ignoring orientation of board.
        '''
        if not self.enabled:
            return VisionRequestResponse(found=False)
        # buoy = self.buoys[srv.target_name]
        if self.est is None:
            return VisionRequestResponse(found=False)
        return VisionRequestResponse(
            pose=PoseStamped(
                header=Header(stamp=self.last_image_time, frame_id='/map'),
                pose=Pose(position=Point(*self.est))),
            found=True)

    def clear_old_observations(self):
        '''
        Observations older than two seconds are discarded.
        '''
        time = rospy.Time.now()
        i = 0
        while i < len(self._times):
            if time - self._times[i] > self.timeout:
                self._times.popleft()
                self._observations.popleft()
                self._pose_pairs.popleft()
            else:
                i += 1
        # print('Clearing')

    def add_observation(self, obs, pose_pair, time):
        '''
        Add a new observation associated with an object
        '''

        self.clear_old_observations()
        # print('Adding...')
        if slen(self._observations) == 0 or np.linalg.norm(
                self._pose_pairs[-1][0] - pose_pair[0]) > self.min_trans:
            self._observations.append(obs)
            self._pose_pairs.append(pose_pair)
            self._times.append(time)

    def get_observations_and_pose_pairs(self):
        '''
        Fetch all recent observations + clear old ones
        '''
        
        self.clear_old_observations()
        return (self._observations, self._pose_pairs)

    def detect(self, c):
        # initialize the shape name and approximate the contour
        target = "unidentified"
        peri = cv2.arcLength(c, True)

        if peri < self.min_contour_area or peri > self.max_contour_area:
            return target
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if self.goal == 'heart':
            if len(approx) == 3:
                target = "Target Aquisition Successful"

            elif len(approx) == 4:
                target = "Partial Target Acquisition"
        else:
            if len(approx) == 1:
                target = "Target Aquisition Successful"
            # TODO check to see what the handle looks like as a contour. 
            elif len(approx) == 2 or len(approx) == 4:
                target = "Partial Target Acquisition"

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
        if self.goal == 'heart':
            lower = np.array(self.heart_lower, dtype="uint8")
            upper = np.array(self.heart_upper, dtype="uint8")
        else:
            lower = np.array(self.handle_lower, dtype="uint8")
            upper = np.array(self.handle_lower, dtype="uint8")

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

                peri = cv2.arcLength(c, True)
                # Grab Largest Contour.
                if peri > peri_max:
                    peri_max = peri
                    max_x = cX
                    max_y = cY
                    m_shape = shape


        if m_shape == "Target Aquisition Successful":
            try:
                self.tf_listener.waitForTransform('/map',
                                                  self.camera_model.tfFrame(),
                                                  self.last_image_time,
                                                  rospy.Duration(0.2))
            except tf.Exception as e:
                rospy.logwarn(
                    "Could not transform camera to map: {}".format(e))
                return False

            (t, rot_q) = self.tf_listener.lookupTransform(
                '/map', self.camera_model.tfFrame(), self.last_image_time)
            R = mil_ros_tools.geometry_helpers.quaternion_matrix(rot_q)
            center = np.array([max_x, max_y])
            self.add_observation(center, (np.array(t), R),
                                 self.last_image_time)

            observations, pose_pairs = self.get_observations_and_pose_pairs()
            if len(observations) > self.min_observations:
                self.est = self.multi_obs.lst_sqr_intersection(
                    observations, pose_pairs)
                self.status = 'Pose found'

            else:
                self.status = '{} observations'.format(len(observations))



def main(args):
    rospy.init_node('drac_ident', anonymous=False)
    DracIdent()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
