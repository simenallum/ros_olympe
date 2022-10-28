#!/usr/bin/python3

import rospy

from std_msgs.msg import Float64, Empty, Bool

from olympe.messages.ardrone3.Piloting import TakeOff, Landing, Emergency, PCMD, moveBy, moveTo
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.skyctrl.CoPilotingState import pilotingSource
from olympe.messages import gimbal, camera, mapper
from olympe.enums.skyctrl.CoPilotingState import PilotingSource_Source

from scipy.spatial.transform import Rotation as R

from olympe_bridge.msg import AttitudeCommand, CameraCommand, MoveByCommand, MoveToCommand


class AnafiBridgeCommandListener:
  def __init__(
        self, 
        anafi, 
        config    
      ) -> None:

    # Initializing reference to connected drone
    self.anafi = anafi

    # Initializing config-parameters
    self.config = config

    # Initializing subscribers
    rospy.Subscriber("/anafi/cmd_takeoff", Empty, self.takeoff_callback)
    rospy.Subscriber("/anafi/cmd_land", Empty, self.land_callback)
    rospy.Subscriber("/anafi/cmd_emergency", Empty, self.emergency_callback)
    rospy.Subscriber("/anafi/cmd_offboard", Bool, self.offboard_callback)
    rospy.Subscriber("/anafi/cmd_rpyt", AttitudeCommand, self.rpyt_callback)
    rospy.Subscriber("/anafi/cmd_moveto", MoveToCommand, self.moveTo_callback)
    rospy.Subscriber("/anafi/cmd_moveby", MoveByCommand, self.moveBy_callback)
    rospy.Subscriber("/anafi/cmd_camera", CameraCommand, self.camera_callback)

    # Initializing publishers
    self.pub_msg_latency = rospy.Publisher("/anafi/msg_latency", Float64, queue_size=1)


  def _bound(
        self, 
        value     : float, 
        value_min : float, 
        value_max : float
      ) -> float:
    return min(max(value, value_min), value_max)
    

  def _bound_percentage(
        self, 
        value : float
      ) -> float:
    return self._bound(value, -100, 100)


  def _switch_manual(self) -> None:
    # Transmitting a single zeroed-attitude command to stabilize the drone
    rpyt_msg = AttitudeCommand()
    rpyt_msg.header.stamp = rospy.Time.now()
    
    rpyt_msg.roll = 0
    rpyt_msg.pitch = 0
    rpyt_msg.yaw = 0
    rpyt_msg.gaz = 0

    self.rpyt_callback(rpyt_msg)
    self.anafi.switch_manual()


  def _switch_offboard(self) -> None:
    # button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
    # axis: 	0 = yaw, 1 = trottle, 2 = roll, 3 = pithch, 4 = camera, 5 = zoom
    if self.anafi.drone.get_state(pilotingSource)["source"] == PilotingSource_Source.SkyController:
      self.anafi.switch_offboard()
    else:
      self._switch_manual()


  def takeoff_callback(self, msg : Empty) -> None:		
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.TakeOff
    self.anafi.drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=10)).wait() 
    rospy.logwarn("Takeoff")


  def land_callback(self, msg : Empty) -> None:
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.Landing		
    self.anafi.drone(Landing()).wait() 
    rospy.loginfo("Land")

  def emergency_callback(self, msg : Empty) -> None:
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.Emergency		
    self.anafi.drone(Emergency()).wait() 
    rospy.logfatal("Emergency!!!")


  def offboard_callback(self, msg : Bool) -> None:
    if msg.data == False:	
      self.anafi.drone.switch_manual()
    else:
      self.anafi.drone.switch_offboard()


  def rpyt_callback(self, msg : AttitudeCommand) -> None:
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.PCMD
  
    time_msg_received = rospy.Time.now()

    # Prioritize to send the command to the drone, before logging the time-delay

    # Negative in pitch to get it into NED
    # Negative in gaz to get it into NED. gaz > 0 means negative velocity downwards

    # Can I just say how much I hate that the PCMD takes the command in yaw, when it clearly is yaw rate...
    self.anafi.drone(
      PCMD( 
        flag=1,
        roll=int(self._bound_percentage(self.config.roll_cmd_scale * msg.roll / self.anafi.max_tilt * 100)),      			# roll [-100, 100] (% of max tilt)
        pitch=int(self._bound_percentage(-self.config.pitch_cmd_scale * msg.pitch / self.anafi.max_tilt * 100)),   		  # pitch [-100, 100] (% of max tilt)
        yaw=int(self._bound_percentage(msg.yaw / self.anafi.max_rotation_speed * 100)), 												        # yaw rate [-100, 100] (% of max yaw rate)
        gaz=int(self._bound_percentage(-self.config.thrust_cmd_scale * msg.gaz / self.anafi.max_vertical_speed * 100)), # vertical speed [-100, 100] (% of max vertical speed)
        timestampAndSeqNum=0
      )
    ) 

    # Logging the time-difference
    time_diff = (time_msg_received - msg.header.stamp).to_sec()
    latency_msg = Float64()
    latency_msg.data = time_diff
    self.pub_msg_latency.publish(latency_msg)


  def moveBy_callback(self, msg : MoveByCommand) -> None:		
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.moveBy
    self.anafi.drone(
      moveBy(
        dX=msg.dx, # displacement along the front axis (m)
        dY=msg.dy, # displacement along the right axis (m)
        dZ=msg.dz, # displacement along the down axis (m)
        dPsi=msg.dyaw # rotation ofthread heading (rad)
      ) >> FlyingStateChanged(state="hovering", _timeout=1)
    ).wait().success()


  def moveTo_callback(self, msg : MoveToCommand) -> None:		
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.moveTo
    self.anafi.drone(moveTo( 
      latitude=msg.latitude, # latitude (degrees)
      longitude=msg.longitude, # longitude (degrees)
      altitude=msg.altitude, # altitude (m)
      heading=msg.heading, # heading relative to the North (degrees)
      orientation_mode=msg.orientation_mode # {TO_TARGET = 1, HEADING_START = 2, HEADING_DURING = 3} 
      ) >> FlyingStateChanged(state="hovering", _timeout=5)
    ).wait().success()


  def camera_callback(self, msg : CameraCommand) -> None:
    if msg.action & 0b001: # take picture
      self.anafi.drone(camera.take_photo(cam_id=0))             # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.take_photo
    if msg.action & 0b010: # start recording
      self.anafi.drone(camera.start_recording(cam_id=0)).wait() # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.start_recording
    if msg.action & 0b100: # stop recording
      self.anafi.drone(camera.stop_recording(cam_id=0)).wait()  # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.stop_recording

    rospy.loginfo("Received camera command")

    # https://developer.parrot.com/docs/olympe/arsdkng_gimbal.html#olympe.messages.gimbal.set_target
    self.anafi.drone(
      gimbal.set_target( 
        gimbal_id=0,
        control_mode='position', # {'position', 'velocity'}
        yaw_frame_of_reference='none',
        yaw=0.0,
        pitch_frame_of_reference=self.anafi.gimbal_frame, # {'absolute', 'relative', 'none'}
        pitch=msg.pitch,
        roll_frame_of_reference=self.anafi.gimbal_frame, # {'absolute', 'relative', 'none'}
        roll=msg.roll
      )
    )
      
    # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.set_zoom_target
    self.anafi.drone(
      camera.set_zoom_target( 
        cam_id=0,
        control_mode='level', # {'level', 'velocity'}
        target=msg.zoom # [1, 3]
      ) 
    ) 


  def run(self) -> None:
    while not rospy.is_shutdown():
      connection = self.anafi.drone.connection_state()
      if connection == False:
        # Lost connection
        rospy.logfatal("Lost connection to the drone")
        break
