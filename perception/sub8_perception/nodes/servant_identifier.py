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
        self.reconfigure_server = DynamicReconfigureServer(ServantIdentifierConfig, self.reconfigure)