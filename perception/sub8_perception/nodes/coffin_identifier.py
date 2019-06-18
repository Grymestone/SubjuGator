#!/usr/bin/env python
from __future__ import print_function
import sys
import tf
import mil_ros_tools
import cv2
import numpy as np
import rospy
from sub8_perception.cfg import CoffinIdentifierConfig
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
This Vampiric Grymoire identifies the the locations of Sleeping Vampires:
    Coffin   --> 2 PVC Containers, One Covered
    Dracula's Body --> PVC Figurine inside of a covered container


In many ways this is 3 perception tasks in one, as we have 3 entities to track here. 

General outline of task
    Find task with Hydrophones.
    Locate open coffin. --> Track center of Bin and have a service call to get it
    Center on coffin.
    Drop the crucifixes.
    Find Dracula. --> Track Dracula handle and have a service call to get it.
    Grab Dracula. --> Get distance to Dracula from DVL (we can assume dvl is hittin him/bottom of bin if we are centered on it.)
    If confident, go to closed coffin.
        Open closed coffin. --> Track coffin handle and have a service call to get it.
        Grab Dracula's twin.
        Surface.
    Else
        Surface.
'''

class VampIdent:

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
        self.reconfigure_server = DynamicReconfigureServer(CoffinIdentifierConfig, self.reconfigure)

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
                self.bin_lower = self.parse_string(config['bin_lower'])
                self.bin_upper = self.parse_string(config['bin_upper'])
                self.handle_lower = self.parse_string(config['handle_lower'])
                self.handle_upper = self.parse_string(config['handle_upper'])
                self.dracula_lower = self.parse_string(config['dracula_lower'])
                self.dracula_upper = self.parse_string(config['dracula_upper'])
                self.min_trans = config['min_trans']
                self.max_velocity = config['max_velocity'] 
                self.timeout = config['timeout']
                self.min_observations = config['min_obs']

            except ValueError as e:
                rospy.logwarn('Invalid dynamic reconfigure: {}'.format(e))
                return self.last_config
                # Dynamic Values for testing
            self.bin_lower = np.array(bin_lower)
            self.bin_upper = np.array(bin_upper)
            self.handle_lower = np.array(handle_lower)
            self.handle_upper = np.array(handle_upper)
            self.dracula_lower = np.array(dracula_lower)
            self.dracula_upper = np.array(dracula_upper)
            self.last_config = config
            rospy.loginfo('Params succesfully updated via dynamic reconfigure')
            return config