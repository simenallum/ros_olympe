#!/usr/bin/env python3

import rospy
import cv2
import math
import os
import queue
import threading
import traceback
import math
import olympe
import numpy as np
import multiprocessing

from scipy.spatial.transform import Rotation as R

from dynamic_reconfigure.server import Server
from olympe_bridge.cfg import setAnafiConfig
from olympe_bridge.msg import AttitudeCommand, CameraCommand, MoveByCommand, MoveToCommand, SkyControllerCommand, Float32Stamped

from anafi_bridge import AnafiBridge
from anafi_listener import AnafiBridgeListener
from anafi_publisher import AnafiBridgePublisher

from dataclasses import dataclass

@dataclass
class AnafiConfig:
  drone_ip : str = ""
  
  roll_cmd_scale : float = 0.0
  pitch_cmd_scale : float = 0.0
  thrust_cmd_scale : float = 0.0

  is_qualisys_available : bool = False
  use_manual_control : bool = False


 
def get_anafi_config_params() -> None:
  anafi_config = AnafiConfig()

  anafi_config.roll_cmd_scale = rospy.get_param("/roll_cmd_scale")
  anafi_config.pitch_cmd_scale = rospy.get_param("/pitch_cmd_scale")
  anafi_config.thrust_cmd_scale = rospy.get_param("/thrust_cmd_scale")

  anafi_config.drone_ip = rospy.get_param("drone_ip")
  anafi_config.is_qualisys_available = rospy.get_param("/qualisys_available")
  anafi_config.use_manual_control = rospy.get_param("/use_manual_control")

  rospy.loginfo("Using scales for roll: {}, pitch: {} and thrust: {}".format(
    anafi_config.roll_cmd_scale,
    anafi_config.pitch_cmd_scale,
    anafi_config.thrust_cmd_scale
    )
  )


def main():
  anafi_bridge = AnafiBridge()	
  
  try:
    anafi_config = get_anafi_config_params()
    drone_ref = anafi_bridge.get_drone_reference()
    
    
    rospy.spin()
  except rospy.ROSInterruptException:
    traceback.print_exc()
    pass


if __name__ == '__main__':
  main()