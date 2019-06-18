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
        self.reconfigure_server = DynamicReconfigureServer(DraculaIdentifierConfig, self.reconfigure)