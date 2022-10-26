#!/usr/bin/python3

import rospy
import os
import threading
import olympe
import numpy as np

from olympe.messages.ardrone3.PilotingSettings import MaxTilt, MaxDistance, MaxAltitude, NoFlyOverMaxDistance, BankedTurn
from olympe.messages.ardrone3.SpeedSettings import MaxVerticalSpeed, MaxRotationSpeed, MaxPitchRollRotationSpeed
from olympe.messages.skyctrl.CoPiloting import setPilotingSource
from olympe.messages.skyctrl.CoPilotingState import pilotingSource
from olympe.messages import gimbal, mapper
from olympe.enums.skyctrl.CoPilotingState import PilotingSource_Source

from dynamic_reconfigure.server import Server
from olympe_bridge.cfg import setAnafiConfig


class Anafi(threading.Thread):
  def __init__(
        self,
        anafi_config
      ) -> None:

    # Initializing node
    # rospy.init_node("anafi_node")
    # self.rate = rospy.Rate(anafi_config.drone_ip)

    # Initializing drone connection
    self.drone_ip = rospy.get_param("drone_ip")
    self.drone = olympe.Drone(anafi_config.drone_ip)
    
    rospy.on_shutdown(self._stop)
    
    # Callback for reconfiguration server
    self.reconfig_server = Server(setAnafiConfig, self._reconfigure_callback)
    
    # Initialize connection to drone
    self._connect_to_drone()
    self._initialize_piloting_source()


  def _connect_to_drone(self) -> None:
    # self.every_event_listener.subscribe()
    
    while True:
      rospy.loginfo("CONNECTING")
      connection = self.drone.connect()
      if connection:
        rospy.loginfo("CONNECTED TO DRONE")
        break
      if rospy.is_shutdown():
        exit()
      rospy.sleep(1)


  def _initialize_piloting_source(self) -> None:
    if self.drone_ip == "192.168.53.1":
      rospy.loginfo("Setting piloting source to Olympe")
      assert self.drone(
        setPilotingSource(source="Controller")
      ).wait().success(), "Failed to set piloting source to Olympe"
    else:
      rospy.logwarn("Piloting source not initialized")


  def _disconnect(self) -> None:
    rospy.loginfo("DISCONNECTING")
    # self.every_event_listener.unsubscribe()
    self.drone.streaming.stop()
    self.drone.disconnect()
    rospy.loginfo("DISCONNECTED")
    

  def _stop(self) -> None:
    rospy.loginfo("AnafiBridge is stopping...")
    self._disconnect()


  def _reconfigure_callback(self, config : setAnafiConfig, level : int) -> setAnafiConfig:
    if level == -1 or level == 1:
      self.drone(MaxTilt(config['max_tilt'])).wait()                                        # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html?#olympe.messages.ardrone3.PilotingSettings.MaxTilt
      self.drone(MaxVerticalSpeed(config['max_vertical_speed'])).wait()											# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxVerticalSpeed
      self.drone(MaxRotationSpeed(config['max_yaw_rotation_speed'])).wait()									# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxRotationSpeed
      self.drone(MaxPitchRollRotationSpeed(config['max_pitch_roll_rotation_speed'])).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxPitchRollRotationSpeed
      self.drone(MaxDistance(config['max_distance'])).wait()																# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.MaxDistance
      self.drone(MaxAltitude(config['max_altitude'])).wait()																# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.MaxAltitude
      self.drone(NoFlyOverMaxDistance(1)).wait()                                            # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.NoFlyOverMaxDistance
      self.drone(BankedTurn(int(config['banked_turn']))).wait()															# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.BankedTurn
      self.max_tilt = np.deg2rad(config['max_tilt'])
      self.max_vertical_speed = config['max_vertical_speed']
      self.max_rotation_speed = np.deg2rad(config['max_yaw_rotation_speed'])
    if level == -1 or level == 2:
      self.gimbal_frame = 'absolute' if config['gimbal_compensation'] else 'relative'
      self.drone(
        gimbal.set_max_speed(
          gimbal_id=0,
          yaw=0, 
          pitch=config['max_gimbal_speed'], # [1 180] (deg/s)
          roll=config['max_gimbal_speed'] # [1 180] (deg/s)
        )
      ).wait()
    return config


  def get_anafi_reference(self):
    return self


  def switch_manual(self) -> None:
    """
    Not perfect, as it will not zero the rpyt-cmds

    Should only be called if truly necessary 
    """

    # button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
    self.drone(mapper.grab(buttons=(0<<0|0<<1|0<<2|1<<3), axes=0)).wait() # bitfields
    self.drone(setPilotingSource(source="SkyController")).wait()
    rospy.loginfo("Control: Manual")


  def switch_offboard(self) -> None:
    """
    Not perfect, as the switch to manual will not zero the rpyt-cmds

    Should only be called if truly necessary 
    """

    # button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
    # axis: 	0 = yaw, 1 = trottle, 2 = roll, 3 = pithch, 4 = camera, 5 = zoom
    if self.drone.get_state(pilotingSource)["source"] == PilotingSource_Source.SkyController:
      self.drone(mapper.grab(buttons=(1<<0|0<<1|1<<2|1<<3), axes=(1<<0|1<<1|1<<2|1<<3|0<<4|0<<5))) # bitfields
      self.drone(setPilotingSource(source="Controller")).wait()
      rospy.loginfo("Control: Offboard")
    else:
      self.switch_manual()


  def run(self):     
    # rate = rospy.Rate(50)
    while not rospy.is_shutdown():
      connection = self.drone.connection_state()
      if connection == False:
        # Lost connection
        rospy.logfatal("Lost connection to the drone")
        break

      # rate.sleep()
