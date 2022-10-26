#!/usr/bin/env python3

import rospy
import traceback
import multiprocessing

import bridge.anafi_drone as anafi_drone
import bridge.anafi_listener as anafi_listener
import bridge.anafi_publisher as anafi_publisher

from dataclasses import dataclass

@dataclass
class AnafiConfig:
  node_rate : int = 1
  drone_ip : str = ""
  
  roll_cmd_scale : float = 0.0
  pitch_cmd_scale : float = 0.0
  thrust_cmd_scale : float = 0.0

  is_qualisys_available : bool = False
  use_manual_control : bool = False

 
def get_anafi_config_params() -> None:
  anafi_config = AnafiConfig()

  anafi_config.node_rate = 100
  anafi_config.drone_ip = rospy.get_param("drone_ip")
  anafi_config.is_qualisys_available = rospy.get_param("/qualisys_available")
  anafi_config.use_manual_control = rospy.get_param("/use_manual_control")

  anafi_config.roll_cmd_scale = rospy.get_param("/roll_cmd_scale")
  anafi_config.pitch_cmd_scale = rospy.get_param("/pitch_cmd_scale")
  anafi_config.thrust_cmd_scale = rospy.get_param("/thrust_cmd_scale")

  rospy.loginfo("Using scales for roll: {}, pitch: {} and thrust: {}".format(
    anafi_config.roll_cmd_scale,
    anafi_config.pitch_cmd_scale,
    anafi_config.thrust_cmd_scale
    )
  )

  return anafi_config


def main():
  rospy.init_node("anafi_bridge")
  anafi_config = get_anafi_config_params()
  
  try:
    anafi = anafi_drone.Anafi(anafi_config)	
    anafi_ref = anafi.get_anafi_reference()
    
    anafi_bridge_listener = anafi_listener.AnafiBridgeListener(anafi_ref, anafi_config)
    anafi_bridge_publisher = anafi_publisher.AnafiBridgePublisher(anafi_ref, anafi_config)

    multiprocessing.Process(target=anafi_bridge_listener.run).start()
    multiprocessing.Process(target=anafi_bridge_publisher.run).start()
    
    rospy.spin()
    
  except rospy.ROSInterruptException:
    traceback.print_exc()
    pass


if __name__ == '__main__':
  main()