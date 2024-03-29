#!/usr/bin/python3

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
import pymap3d
import sys

from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Int8, UInt8, UInt16, Int8, Float64, String, Header, Time, Empty, Bool
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, TwistStamped, Vector3Stamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, NavSatFix, NavSatStatus

from olympe.messages.ardrone3.Piloting import TakeOff, Landing, Emergency, PCMD, moveBy, moveTo
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, SpeedChanged, AttitudeChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt, MaxDistance, MaxAltitude, NoFlyOverMaxDistance, BankedTurn
from olympe.messages.ardrone3.PilotingSettingsState import MaxTiltChanged, MaxDistanceChanged, MaxAltitudeChanged, NoFlyOverMaxDistanceChanged, BankedTurnChanged
from olympe.messages.ardrone3.SpeedSettings import MaxVerticalSpeed, MaxRotationSpeed, MaxPitchRollRotationSpeed
from olympe.messages.ardrone3.SpeedSettingsState import MaxVerticalSpeedChanged, MaxRotationSpeedChanged, MaxPitchRollRotationSpeedChanged

from olympe.messages.skyctrl.CoPiloting import setPilotingSource

from olympe.messages import gimbal, camera, mapper
from olympe.enums.mapper import button_event

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from dynamic_reconfigure.server import Server
from olympe_bridge.cfg import setAnafiConfig
# from olympe_bridge.msg import AttitudeCommand, CameraCommand, MoveByCommand, MoveToCommand, SkyControllerCommand, Float32Stamped
from anafi_uav_msgs.msg import AttitudeCommand, CameraCommand, MoveByCommand, MoveToCommand, SkyControllerCommand, Float32Stamped


olympe.log.update_config({"loggers": {"olympe": {"level": "ERROR"}}})

DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT", "554")
GNSS_ORIGIN_DRONE_LAB = (63.418215, 10.401655, 0) # origin of NED, setting origin to be @ the drone lab at NTNU

class Anafi(threading.Thread):
  def __init__(self):
    self.roll_cmd_scale = rospy.get_param("/roll_cmd_scale")
    self.pitch_cmd_scale = rospy.get_param("/pitch_cmd_scale")
    self.thrust_cmd_scale = rospy.get_param("/thrust_cmd_scale")

    rospy.loginfo("Using scales for roll: {}, pitch: {} and thrust: {}".format(
      self.roll_cmd_scale,
      self.pitch_cmd_scale,
      self.thrust_cmd_scale
      )
    )

    self.ned_origo_in_lla : np.ndarray = None

    self.is_qualisys_available = rospy.get_param("/qualisys_available")
    self.is_sim = rospy.get_param("/is_sim")
    self.use_manual_control = rospy.get_param("/use_manual_control")

    if self.is_qualisys_available:			
      rospy.loginfo("Flying at drone lab: Qualisys is available")
    else:
      if self.is_sim:
        rospy.loginfo("Flying in the simulator")
      else:
        rospy.loginfo("Flying a real drone outside the lab: Qualisys is not available")
          
    self.pub_image = rospy.Publisher("/anafi/image", Image, queue_size=1)
    self.pub_time = rospy.Publisher("/anafi/time", Time, queue_size=1)
    self.pub_attitude = rospy.Publisher("/anafi/attitude", QuaternionStamped, queue_size=1)
    self.pub_gnss_location = rospy.Publisher("/anafi/gnss_location", NavSatFix, queue_size=1)
    self.pub_height = rospy.Publisher("/anafi/height", Float32Stamped, queue_size=1)
    self.pub_optical_flow_velocities = rospy.Publisher("/anafi/optical_flow_velocities", Vector3Stamped, queue_size=1)
    self.pub_link_goodput = rospy.Publisher("/anafi/link_goodput", UInt16, queue_size=1)
    self.pub_link_quality = rospy.Publisher("/anafi/link_quality", UInt8, queue_size=1)
    self.pub_wifi_rssi = rospy.Publisher("/anafi/wifi_rssi", Int8, queue_size=1)
    self.pub_battery = rospy.Publisher("/anafi/battery", Float64, queue_size=1)
    self.pub_state = rospy.Publisher("/anafi/state", String, queue_size=1)
    self.pub_mode = rospy.Publisher("/anafi/mode", String, queue_size=1)
    self.pub_pose = rospy.Publisher("/anafi/pose", PoseStamped, queue_size=1)
    self.pub_odometry = rospy.Publisher("/anafi/odometry", Odometry, queue_size=1)
    self.pub_rpy = rospy.Publisher("/anafi/rpy", Vector3Stamped, queue_size=1)
    self.pub_skycontroller = rospy.Publisher("/skycontroller/command", SkyControllerCommand, queue_size=1)
    self.pub_polled_velocities = rospy.Publisher("/anafi/polled_body_velocities", TwistStamped, queue_size=1)
    self.pub_msg_latency = rospy.Publisher("/anafi/msg_latency", Float64, queue_size=1)
    self.pub_ned_pos_from_gnss = rospy.Publisher("/anafi/ned_pos_from_gnss", PointStamped, queue_size=1)
    self.pub_ned_frame_gnss_origin = rospy.Publisher("/anafi/ned_frame_gnss_origin", Vector3Stamped, queue_size=1)

    rospy.Subscriber("/anafi/cmd_takeoff", Empty, self._takeoff_callback)
    rospy.Subscriber("/anafi/cmd_land", Empty, self._land_callback)
    rospy.Subscriber("/anafi/cmd_emergency", Empty, self._emergency_callback)
    rospy.Subscriber("/anafi/cmd_offboard", Bool, self._set_control_source_callback)
    rospy.Subscriber("/anafi/cmd_rpyt", AttitudeCommand, self._rpyt_callback)
    rospy.Subscriber("/anafi/cmd_moveto", MoveToCommand, self._moveTo_callback)
    rospy.Subscriber("/anafi/cmd_moveby", MoveByCommand, self._moveBy_callback)
    rospy.Subscriber("/anafi/cmd_camera", CameraCommand, self._camera_callback)
    rospy.Subscriber("/anafi/cmd_moveto_ned_position", PointStamped, self._move_to_ned_pos_cb)

    # Prefer service for land / takeoff, as topics will likely be duplicated when interfacing with ROS2
    # Topics risks sending the takeoff / land commands multiple times, resulting in the drone oscillating
    # between landings and takeoffs...
    # The topics are left, in case they could be useful. The reader should beware before use, though
    rospy.Service("/anafi/request_takeoff", Trigger, self._request_takeoff_cb)
    rospy.Service("/anafi/request_land", Trigger, self._request_land_cb)		

    if self.is_qualisys_available:
      rospy.Subscriber("/qualisys/Anafi/pose_downsampled", PoseStamped, self.qualisys_callback)
      self.last_received_location = NavSatFix()
    
    self.drone_ip = rospy.get_param("drone_ip")
    self.drone = olympe.Drone(self.drone_ip)
    
    # Create listener for RC events
    self.every_event_listener = EveryEventListener(self.drone, self)
    self.every_event_listener.subscribe()

    rospy.on_shutdown(self._stop)
    
    self.srv = Server(setAnafiConfig, self._reconfigure_callback)

    self._connect()
    
    # To convert OpenCV images to ROS images
    self.bridge = CvBridge()
    

  def _connect(self) -> None:
    self.every_event_listener.subscribe()
    
    rate = rospy.Rate(1) # 1hz
    while True:
      self.pub_state.publish("CONNECTING")
      connection = self.drone.connect()
      if connection:
        rospy.loginfo("CONNECTED TO DRONE")
        break
      if rospy.is_shutdown():
        exit()
      rate.sleep()

    if self.drone_ip == "192.168.53.1":
      if self.use_manual_control:
        rospy.loginfo("Setting manual control")
        assert self._switch_to_manual_control(), "Unable to set manual control. Exiting"
      else:
        rospy.loginfo("Setting piloting source to Olympe")
        assert self._switch_to_olympe_control(), "Unable to set control to Olympe. Exiting"
    else:
      rospy.logwarn("Piloting source not initialized. This should only occur in the simulator...")

    self.frame_queue = queue.Queue()
    self.flush_queue_lock = threading.Lock()

    if DRONE_RTSP_PORT is not None:
      self.drone.streaming.server_addr = rospy.get_param("drone_ip") + f":{DRONE_RTSP_PORT}"

    # Setup the callback functions to do some live video processing
    self.drone.streaming.set_callbacks(
      raw_cb=self._yuv_frame_cb,
      flush_raw_cb=self._flush_cb
    )

    self.drone.streaming.start()
    

  def _disconnect(self) -> None:
    self.pub_state.publish("DISCONNECTING")
    self.every_event_listener.unsubscribe()
    self.drone.streaming.stop()
    self.drone.disconnect()
    self.pub_state.publish("DISCONNECTED")
    

  def _stop(self) -> None:
    rospy.loginfo("AnafiBridge is stopping...")
    self._disconnect()


  def _reconfigure_callback(
        self, 
        config : setAnafiConfig, 
        level  : int
      ) -> setAnafiConfig:
    if level == -1 or level == 1:
      self.drone(MaxTilt(config['max_tilt'])).wait() 																				# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html?#olympe.messages.ardrone3.PilotingSettings.MaxTilt
      self.drone(MaxVerticalSpeed(config['max_vertical_speed'])).wait() 										# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxVerticalSpeed
      self.drone(MaxRotationSpeed(config['max_yaw_rotation_speed'])).wait() 								# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxRotationSpeed
      self.drone(MaxPitchRollRotationSpeed(config['max_pitch_roll_rotation_speed'])).wait() # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.SpeedSettings.MaxPitchRollRotationSpeed
      self.drone(MaxDistance(config['max_distance'])).wait() 																# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.MaxDistance
      self.drone(MaxAltitude(config['max_altitude'])).wait() 																# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.MaxAltitude
      self.drone(NoFlyOverMaxDistance(1)).wait() 																						# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.NoFlyOverMaxDistance
      self.drone(BankedTurn(int(config['banked_turn']))).wait() 														# https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.PilotingSettings.BankedTurn
      self.max_tilt = np.deg2rad(config['max_tilt'])
      self.max_vertical_speed = config['max_vertical_speed']
      self.max_rotation_speed = np.deg2rad(config['max_yaw_rotation_speed'])
    if level == -1 or level == 2:
      self.gimbal_frame = 'absolute' if config['gimbal_compensation'] else 'relative'
      self.drone(gimbal.set_max_speed(
        gimbal_id=0,
        yaw=0, 
        pitch=config['max_gimbal_speed'], # [1 180] (deg/s)
        roll=config['max_gimbal_speed'] # [1 180] (deg/s)
        )).wait()
    return config
    

  def _yuv_frame_cb(self, yuv_frame) -> None:  
    yuv_frame.ref()
    self.frame_queue.put_nowait(yuv_frame)


  def _flush_cb(self, _) -> Bool:
    with self.flush_queue_lock:
      while not self.frame_queue.empty():
        self.frame_queue.get_nowait().unref()
    return True


  def _yuv_callback(self, yuv_frame) -> None:
    # Use OpenCV to convert the yuv frame to RGB
    # the VideoFrame.info() dictionary contains some useful information
    # such as the video resolution
    info = yuv_frame.info()
    height, width = (  # noqa
      info["raw"]["frame"]["info"]["height"],
      info["raw"]["frame"]["info"]["width"],
    )

    # convert pdraw YUV flag to OpenCV YUV flag
    cv2_cvt_color_flag = {
      olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
      olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
    }[yuv_frame.format()]

    # yuv_frame.as_ndarray() is a 2D numpy array with the proper "shape"
    # i.e (3 * height / 2, width) because it's a YUV I420 or NV12 frame

    # Use OpenCV to convert the yuv frame to RGB
    cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)  # noqa
    
    # Publish image
    msg_image = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
    self.pub_image.publish(msg_image)

    # yuv_frame.vmeta() returns a dictionary that contains additional metadata from the drone (GPS coordinates, battery percentage, ...)
    metadata = yuv_frame.vmeta()
    rospy.logdebug_throttle(10, "yuv_frame.vmeta = " + str(metadata))
        
    if metadata[1] != None:
      header = Header()
      header.stamp = rospy.Time.now()
      header.frame_id = '/body'
    
      frame_timestamp = info['raw']['frame']['timestamp'] # timestamp [microsec]
      secs = int(frame_timestamp // 1e6)
      microsecs = (frame_timestamp - (1e6 * secs))
      nanosecs = microsecs * 1e3
      msg_time = Time()
      msg_time.data = rospy.Time(secs, nanosecs) # secs = int(frame_timestamp//1e6), nsecs = int(frame_timestamp%1e6*1e3)
      self.pub_time.publish(msg_time)

      if "drone" in metadata[1]:
        drone_data = metadata[1]['drone']

        drone_quat = drone_data['quat'] # attitude
        quat = [drone_quat['x'], drone_quat['y'], drone_quat['z'], drone_quat['w']]

        Rot = R.from_quat(quat)
        drone_rpy = Rot.as_euler('xyz', degrees=False)

        # TODO Rewrite this part to more readable code. Currenly using magic number etc.
        # This part adds the offsets to the quaternions before it is published
        if self.drone_ip == "192.168.53.1":
          # Real drone
          correction_terms = (-0.015576460455065291, -0.012294590577876349, 0)
        else:
          # Simulator
          correction_terms = (-0.0002661987518324087, -0.0041069024624204534, 0)

        drone_rpy_corrected = drone_rpy + correction_terms
        rot_corrected_body_to_vehicle = R.from_euler('xyz', drone_rpy_corrected, degrees=False)
        quat = R.as_quat(rot_corrected_body_to_vehicle)

        msg_attitude = QuaternionStamped()
        msg_attitude.header = header
        msg_attitude.quaternion = Quaternion(quat[0], quat[1], quat[2], quat[3])
        self.pub_attitude.publish(msg_attitude)

        msg_location = NavSatFix()
        status = NavSatStatus()
        if "location" in drone_data: # Sometimes unavailable?
          location = drone_data['location']       # GNSS location [500.0=not available] (decimal deg) 500 what??
          if location != {}:			
            msg_location.header = header
            msg_location.header.frame_id = 'world'
            msg_location.latitude = location['latitude']            # [deg]
            msg_location.longitude = location['longitude']          # [deg]
            msg_location.altitude = location['altitude_egm96amsl']  # [m] using EGM96 geoid (!!)

            status.status = 0 	# Position fix

        elif self.is_qualisys_available:
          msg_location.header = header
          msg_location.header.frame_id = 'world'
          msg_location.latitude = self.last_received_location.latitude
          msg_location.longitude = self.last_received_location.longitude
          msg_location.altitude = self.last_received_location.altitude

          status.status = 0	# Position fix

        else:
          # No connection - setting to zero to make it explicit
          msg_location.latitude = 0   # [deg]
          msg_location.longitude = 0  # [deg]
          msg_location.altitude = 0   # [m] over WGS84

          status.status = -1	# No position fix

        if status.status == 0:

          n, e, d = self._calculate_ned_position_wgs84(msg_location) # WGS84 assumed OK in the calculations !!

          max_distance = 4000 # [m]
          if np.any(np.array([n, e, d]) > max_distance):
            rospy.logwarn_throttle(1, "NED values exceed maximum distance of {} meters".format(max_distance)) 

          msg_ned_pos_from_gnss = PointStamped()
          msg_ned_pos_from_gnss.header = header 
          msg_ned_pos_from_gnss.point.x = n
          msg_ned_pos_from_gnss.point.y = e 
          msg_ned_pos_from_gnss.point.z = d 

          self.pub_ned_pos_from_gnss.publish(msg_ned_pos_from_gnss)
          
          # Publish the origin for the GNSS frame. Needed in some other nodes for debugging.
          msg_ned_frame_gnss_origin = Vector3Stamped()
          msg_ned_frame_gnss_origin.header = header
          msg_ned_frame_gnss_origin.vector.x = self.ned_origo_in_lla[0]
          msg_ned_frame_gnss_origin.vector.y = self.ned_origo_in_lla[1]
          msg_ned_frame_gnss_origin.vector.z = self.ned_origo_in_lla[2]
          self.pub_ned_frame_gnss_origin.publish(msg_ned_frame_gnss_origin)

        msg_location.status = status
        self.pub_gnss_location.publish(msg_location)

        state = drone_data['flying_state'] # ['LANDED', 'MOTOR_RAMPING', 'TAKINGOFF', 'HOVERING', 'FLYING', 'LANDING', 'EMERGENCY']
        self.pub_state.publish(state)

        ground_distance = drone_data['ground_distance'] # barometer (m) 

        if state not in ["FS_LANDED", "FS_MOTOR_RAMPING", "FS_TAKINGOFF"]: # -> Reads wrong data during these states
          height_msg = Float32Stamped()
          height_msg.header = header
          height_msg.data = ground_distance
            
          self.pub_height.publish(height_msg)

        of_speed = drone_data['speed'] # opticalflow speed (m/s)
        rotation_matrix_body_to_vehicle = rot_corrected_body_to_vehicle.as_matrix()   

        of_cam_north_dot = of_speed['north']
        of_cam_east_dot = of_speed['east']
        of_cam_down_dot = of_speed['down']

        # Might become somewhat inaccurate if the yaw-estimate is poor
        optical_flow_velocities_body = rotation_matrix_body_to_vehicle.T @ np.array([[of_cam_north_dot], [of_cam_east_dot], [of_cam_down_dot]], dtype=float) 

        msg_speed = Vector3Stamped()
        msg_speed.header = header
        msg_speed.header.frame_id = 'body'
        msg_speed.vector.x = optical_flow_velocities_body[0]
        msg_speed.vector.y = optical_flow_velocities_body[1]
        msg_speed.vector.z = optical_flow_velocities_body[2]
        self.pub_optical_flow_velocities.publish(msg_speed)

        msg_pose = PoseStamped()
        msg_pose.header = header
        msg_pose.pose.position.x = msg_location.latitude
        msg_pose.pose.position.y = msg_location.longitude
        msg_pose.pose.position.z = ground_distance
        msg_pose.pose.orientation = msg_attitude.quaternion
        self.pub_pose.publish(msg_pose)

        msg_odometry = Odometry()
        msg_odometry.header = header
        msg_odometry.child_frame_id = 'body'
        msg_odometry.pose.pose = msg_pose.pose
        msg_odometry.twist.twist.linear.x =  math.cos(drone_rpy[2])*msg_speed.vector.x + math.sin(drone_rpy[2])*msg_speed.vector.y
        msg_odometry.twist.twist.linear.y = -math.sin(drone_rpy[2])*msg_speed.vector.x + math.cos(drone_rpy[2])*msg_speed.vector.y
        msg_odometry.twist.twist.linear.z = msg_speed.vector.z
        self.pub_odometry.publish(msg_odometry)

        battery_percentage = drone_data['battery_percentage'] # [0 = empty, 100 = full]
        
        battery_percentage_msg = Float64() # Using Int8 or UInt8 causes confusion with char over the ROS1 - ROS2 bridge
        battery_percentage_msg.data = battery_percentage
        self.pub_battery.publish(battery_percentage_msg)

        # Log battery percentage
        if battery_percentage >= 30:
          if battery_percentage % 10 == 0:
            rospy.loginfo_throttle(100, "Battery level: " + str(battery_percentage) + "%")
        else:
          if battery_percentage >= 20:
            rospy.logwarn_throttle(10, "Low battery: " + str(battery_percentage) + "%")
          else:
            if battery_percentage >= 10:
              rospy.logerr_throttle(1, "Critical battery: " + str(battery_percentage) + "%")
            else:
              rospy.logfatal_throttle(0.1, "Empty battery: " + str(battery_percentage) + "%")		

      if 'links' in metadata[1]:
        link_data = metadata[1]['links']
        wifi_data = link_data[0]['wifi']
        link_goodput = wifi_data['goodput'] # throughput of the connection (b/s)
        self.pub_link_goodput.publish(link_goodput)

        link_quality = wifi_data['quality'] # [0=bad, 5=good]
        self.pub_link_quality.publish(link_quality)

        wifi_rssi = wifi_data['rssi'] # signal strength [-100=bad, 0=good] (dBm)
        self.pub_wifi_rssi.publish(wifi_rssi)
      
        # Log signal strength
        if wifi_rssi <= -60:
          if wifi_rssi >= -70:
            rospy.loginfo_throttle(100, "Signal strength: " + str(wifi_rssi) + "dBm")
          else:
            if wifi_rssi >= -80:
              rospy.logwarn_throttle(10, "Weak signal: " + str(wifi_rssi) + "dBm")
            else:
              if wifi_rssi >= -90:
                rospy.logerr_throttle(1, "Unreliable signal:" + str(wifi_rssi) + "dBm")
              else:
                rospy.logfatal_throttle(0.1, "Unusable signal: " + str(wifi_rssi) + "dBm")
    else:
      rospy.logwarn("Packet lost!")


  def _request_takeoff_cb(self, request : TriggerRequest) -> TriggerResponse:
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.TakeOff
    rospy.logwarn("Takeoff requested")
    self.drone(TakeOff()) 
    rospy.logwarn("Takeoff")
    return TriggerResponse(True, "")


  def _request_land_cb(self, request : TriggerRequest) -> TriggerResponse:
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.Landing		
    rospy.logwarn("Land requested")
    self.drone(Landing()) 
    rospy.logwarn("Land")
    return TriggerResponse(True, "")


  def _takeoff_callback(self, msg : Empty) -> None:		
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.TakeOff
    rospy.logwarn("Takeoff command received")
    self.drone(TakeOff()) # Drop wait. Better to send a new message instead of blocking the thread
    rospy.logwarn("Takeoff")


  def _land_callback(self, msg : Empty) -> None:
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.Landing		
    rospy.logwarn("Land command received")
    self.drone(Landing()) # Drop wait. Better to send a new message instead of blocking the thread
    rospy.logwarn("Land")


  def _emergency_callback(self, msg) -> None:
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.Emergency		
    self.drone(Emergency()) 
    rospy.logfatal("Emergency!!!")


  def _set_control_source_callback(self, msg) -> None:
    if msg.data == False:	
      self._switch_to_manual_control()
    else:
      self._switch_to_olympe_control()


  def _rpyt_callback(self, msg : AttitudeCommand) -> None:
    time_msg_received = rospy.Time.now()

    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.PCMD
    # Testing in the simulator shows that the roll and pitch-commands must be multiplied with 0.5
    # Negative in pitch to get it into NED
    # Negative in gaz to get it into NED. gaz > 0 means negative velocity downwards
    # Can I just say how much I hate that the PCMD takes the command in yaw, when it clearly is yaw rate

    self.drone(PCMD( 
      flag=1,
      roll=int(self._bound_percentage(self.roll_cmd_scale * msg.roll / self.max_tilt * 100)),      			# roll [-100, 100] (% of max tilt)
      pitch=int(self._bound_percentage(-self.pitch_cmd_scale * msg.pitch / self.max_tilt * 100)),   			# pitch [-100, 100] (% of max tilt)
      yaw=int(self._bound_percentage(msg.yaw / self.max_rotation_speed * 100)), 													# yaw rate [-100, 100] (% of max yaw rate)
      gaz=int(self._bound_percentage(-self.thrust_cmd_scale * msg.gaz / self.max_vertical_speed * 100)), # vertical speed [-100, 100] (% of max vertical speed)
      timestampAndSeqNum=0)) # for debug only

    time_diff = (time_msg_received - msg.header.stamp).to_sec()
    latency_msg = Float64()
    latency_msg.data = time_diff
    self.pub_msg_latency.publish(latency_msg)


  def _moveBy_callback(self, msg : MoveByCommand) -> None:		
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.moveBy

    moveby_str = "Move by (x, y, z) (" + str(msg.dx) + ", " + str(msg.dy) + ", " + str(msg.dz) + ")"
    rospy.logwarn(moveby_str)

    self.drone(moveBy(
      dX=msg.dx, # displacement along the front axis (m)
      dY=msg.dy, # displacement along the right axis (m)
      dZ=msg.dz, # displacement along the down axis (m)
      dPsi=msg.dyaw # rotation of heading (rad)
      ) 
    )


  def _moveTo_callback(self, msg : MoveToCommand) -> None:		
    # https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html#olympe.messages.ardrone3.Piloting.moveTo

    moveto_str = "Move to (latitude, longitude, altitude) (" + str(msg.latitude) + ", " + str(msg.longitude) + ", " + str(msg.altitude) + ")"
    rospy.logwarn(moveto_str)

    self.drone(moveTo( 
      latitude=msg.latitude, # latitude (degrees)
      longitude=msg.longitude, # longitude (degrees)
      altitude=msg.altitude, # altitude (m)
      heading=msg.heading, # heading relative to the North (degrees)
      orientation_mode=msg.orientation_mode # {TO_TARGET = 1, HEADING_START = 2, HEADING_DURING = 3} 
      )
    )


  def _move_to_ned_pos_cb(self, msg : PointStamped) -> None:
    if self.ned_origo_in_lla is None:
      rospy.logerr_throttle(1, "GNSS-measurements are not received, such that the origin is not initialized. Cannot move to desired NED-position")
      return 

    if msg.point.z > -0.5:
      rospy.logwarn("Desired altitude is {} m above initial point. Risk of crash. Aborting...".format(-msg.point.z))
      return

    ell_wgs84 = pymap3d.Ellipsoid('wgs84')

    lat_0 = self.ned_origo_in_lla[0]
    lon_0 = self.ned_origo_in_lla[1]
    a_0 = self.ned_origo_in_lla[2]

    lat, lon, a = pymap3d.ned2geodetic( # May not be accurate enough for precise positioning...
      msg.point.x, 
      msg.point.y, 
      msg.point.z, 
      lat_0, 
      lon_0, 
      a_0, 
      ell=ell_wgs84, 
      deg=True
    )
    
    self.drone(moveTo( 
      latitude=lat, # latitude (degrees)
      longitude=lon, # longitude (degrees)
      altitude=a, # altitude (m)
      heading=0, # heading relative to the North (degrees)
      orientation_mode=3 # {TO_TARGET = 1, HEADING_START = 2, HEADING_DURING = 3} 
      ) 
    )


  def _camera_callback(self, msg) -> None:
    if msg.action & 0b001: # take picture
      self.drone(camera.take_photo(cam_id=0))             # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.take_photo
    if msg.action & 0b010: # start recording
      self.drone(camera.start_recording(cam_id=0)).wait() # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.start_recording
    if msg.action & 0b100: # _stop recording
      self.drone(camera.stop_recording(cam_id=0)).wait()  # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.stop_recording

    rospy.loginfo("Received gimbal command")

    # https://developer.parrot.com/docs/olympe/arsdkng_gimbal.html#olympe.messages.gimbal.set_target
    self.drone(gimbal.set_target( 
      gimbal_id=0,
      control_mode='position', # {'position', 'velocity'}
      yaw_frame_of_reference='none',
      yaw=0.0,
      pitch_frame_of_reference=self.gimbal_frame, # {'absolute', 'relative', 'none'}
      pitch=msg.pitch,
      roll_frame_of_reference=self.gimbal_frame, # {'absolute', 'relative', 'none'}
      roll=msg.roll))
      
    # https://developer.parrot.com/docs/olympe/arsdkng_camera.html#olympe.messages.camera.set_zoom_target
    self.drone(camera.set_zoom_target( 
      cam_id=0,
      control_mode='level', # {'level', 'velocity'}
      target=msg.zoom)) # [1, 3]
      

  def _bound(
        self, 
        value     : float, 
        value_min : float, 
        value_max : float 
      ) -> float:
    return min(max(value, value_min), value_max)
    

  def _bound_percentage(self, value : float) -> float:
    return self._bound(value, -100, 100)


  def _switch_to_manual_control(self) -> bool:
    msg_rpyt = SkyControllerCommand()
    msg_rpyt.header.stamp = rospy.Time.now()
    msg_rpyt.header.frame_id = '/body'
    self.pub_skycontroller.publish(msg_rpyt)

    # button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
    self.drone(mapper.grab(buttons=(0<<0|0<<1|0<<2|1<<3), axes=0)).wait() # bitfields
    if not self.drone(setPilotingSource(source="SkyController")).wait():
      rospy.logerr("Failed to set manual control")
      return False
    rospy.loginfo("Control: Manual")
    return True


  def _switch_to_olympe_control(self) -> bool: 
    # button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
    # axis: 	0 = yaw, 1 = trottle, 2 = roll, 3 = pithch, 4 = camera, 5 = zoom
    self.drone(mapper.grab(buttons=(1<<0|0<<1|1<<2|1<<3), axes=(1<<0|1<<1|1<<2|1<<3|0<<4|0<<5))) # bitfields
    if not self.drone(setPilotingSource(source="Controller")).wait():
      rospy.logerr("Failed to set control to olympe")
      return False
    rospy.loginfo("Control: Olympe")
    return True


  def qualisys_callback(self, msg: PoseStamped) -> None:
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z

    noise = np.random.normal(0, 0.05, 3) # Zero mean, 0.05 std. dev gaussian noise

    x += noise[0]
    y += noise[1]
    z += noise[2]

    ell_wgs84 = pymap3d.Ellipsoid('wgs84')
    lat0, lon0, h0 = GNSS_ORIGIN_DRONE_LAB   

    lat1, lon1, h1 = pymap3d.ned2geodetic(x, y, z, \
                      lat0, lon0, h0, \
                      ell=ell_wgs84, deg=True)  # wgs84 ellisoid

    self.last_received_location.latitude = lat1
    self.last_received_location.longitude = lon1
    self.last_received_location.altitude = h1


  def _calculate_ned_position_wgs84(self, gnss_msg : NavSatFix) -> np.ndarray:
    warn_msg_str = """\n
    Calculation of NED-position is outdated, as it relies on the WGS84 ellipsoid, and not the
    updated EGM96 version. Thus, the altitude between the WGS84 and the
    geoid will be different. Either find a good conversion, or use the
    altitudal values with great care.\n
    Hypothesis: The altitude difference between the geoid and the WGS84
    should be negligable for our use case. As long as the altitude is used
    to generate the NED-position, it should have little impact...\n
    """
    rospy.logwarn_once(warn_msg_str)

    ell_wgs84 = pymap3d.Ellipsoid('wgs84')
    lat, lon, a = gnss_msg.latitude, gnss_msg.longitude, gnss_msg.altitude

    if self.ned_origo_in_lla is None:
      if self.is_qualisys_available:
        self.ned_origo_in_lla = (lat, lon, a) # This has to be commented out for testing platform tracker
        # self.ned_origo_in_lla = GNSS_ORIGIN_DRONE_LAB # This has to be commented out for testing pix2geo
      else:
        self.ned_origo_in_lla = (lat, lon, a)

    lat_0 = self.ned_origo_in_lla[0]
    lon_0 = self.ned_origo_in_lla[1]
    a_0 = self.ned_origo_in_lla[2]

    return pymap3d.geodetic2ned(lat, lon, a, lat_0, lon_0, a_0, ell=ell_wgs84, deg=True)


  def run(self) -> None: 
    freq = 100
    rate = rospy.Rate(freq) # 100hz
    
    rospy.logdebug('MaxTilt = %f [%f, %f]', self.drone.get_state(MaxTiltChanged)["current"], self.drone.get_state(MaxTiltChanged)["min"], self.drone.get_state(MaxTiltChanged)["max"])
    rospy.logdebug('MaxVerticalSpeed = %f [%f, %f]', self.drone.get_state(MaxVerticalSpeedChanged)["current"], self.drone.get_state(MaxVerticalSpeedChanged)["min"], self.drone.get_state(MaxVerticalSpeedChanged)["max"])
    rospy.logdebug('MaxRotationSpeed = %f [%f, %f]', self.drone.get_state(MaxRotationSpeedChanged)["current"], self.drone.get_state(MaxRotationSpeedChanged)["min"], self.drone.get_state(MaxRotationSpeedChanged)["max"])
    rospy.logdebug('MaxPitchRollRotationSpeed = %f [%f, %f]', self.drone.get_state(MaxPitchRollRotationSpeedChanged)["current"], self.drone.get_state(MaxPitchRollRotationSpeedChanged)["min"], self.drone.get_state(MaxPitchRollRotationSpeedChanged)["max"])
    rospy.logdebug('MaxDistance = %f [%f, %f]', self.drone.get_state(MaxDistanceChanged)["current"], self.drone.get_state(MaxDistanceChanged)["min"], self.drone.get_state(MaxDistanceChanged)["max"])
    rospy.logdebug('MaxAltitude = %f [%f, %f]', self.drone.get_state(MaxAltitudeChanged)["current"], self.drone.get_state(MaxAltitudeChanged)["min"], self.drone.get_state(MaxAltitudeChanged)["max"])
    rospy.logdebug('NoFlyOverMaxDistance = %i', self.drone.get_state(NoFlyOverMaxDistanceChanged)["shouldNotFlyOver"])
    rospy.logdebug('BankedTurn = %i', self.drone.get_state(BankedTurnChanged)["state"])
    
    i = 0
    min_iterations = 5 # Should techically be 20, but higher update freq should not matter  
    prev_attitude = None

    while not rospy.is_shutdown():
      connection = self.drone.connection_state()
      if connection == False:
        rospy.logfatal("Lost connection to the Anafi")
        self._disconnect()
        self._connect()

      if i >= min_iterations:
        att_euler = self.drone.get_state(AttitudeChanged)

        roll = att_euler["roll"]     
        pitch = att_euler["pitch"]    
        yaw = att_euler["yaw"]

        msg_rpy = Vector3Stamped()
        msg_rpy.header.stamp = rospy.Time.now()
        msg_rpy.header.frame_id = 'world'
        msg_rpy.vector.x = roll
        msg_rpy.vector.y = pitch
        msg_rpy.vector.z = yaw
        self.pub_rpy.publish(msg_rpy)

        # Check if new 5 Hz data has arrived
        if prev_attitude is None or prev_attitude != [roll, pitch, yaw]:
          prev_attitude = [roll, pitch, yaw]

          # Check if new telemetry
          pos_dot_ned = self.drone.get_state(SpeedChanged)
          velocity_ned = np.array(
            [
              [pos_dot_ned["speedX"]], 
              [pos_dot_ned["speedY"]], 
              [pos_dot_ned["speedZ"]]
            ]
          )

          def Rx(radians):
            c = np.cos(radians)
            s = np.sin(radians)

            return np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s, c]])

          def Ry(radians):
            c = np.cos(radians)
            s = np.sin(radians)

            return np.array([[c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]])

          def Rz(radians):
            c = np.cos(radians)
            s = np.sin(radians)

            return np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])

          # Tried to use scipy's rotaion-matrix, but wouldn't get quite right...
          # R_scipy = R.from_euler('zyx', [roll, pitch, yaw], degrees=False).as_matrix()
          R_ned_to_body = Rx(roll).T @ Ry(pitch).T @ Rz(yaw).T

          velocity_body = R_ned_to_body @ velocity_ned

          twist_stamped = TwistStamped()
          twist_stamped.header.stamp = rospy.Time.now()
          twist_stamped.twist.linear.x = velocity_body[0]
          twist_stamped.twist.linear.y = velocity_body[1]
          twist_stamped.twist.linear.z = velocity_body[2]
          self.pub_polled_velocities.publish(twist_stamped)
          
          i = 0
      else:
        i += 1

      with self.flush_queue_lock:
        try:					
          yuv_frame = self.frame_queue.get(timeout=0.01)
        except queue.Empty:
          continue
        
        try:
          self._yuv_callback(yuv_frame)
        except Exception:
          # Continue popping frame from the queue even if it fails to show one frame
          traceback.print_exc()
          continue
        finally:
          # Unref the yuv frame to avoid starving the video buffer pool
          yuv_frame.unref()
                
      rate.sleep()


class EveryEventListener(olympe.EventListener):
  def __init__(self, drone, anafi):
    self.drone = drone
    self.anafi = anafi
        
    self.msg_rpyt = SkyControllerCommand()
    
    super().__init__(drone)


  def print_event(self, event): # Serializes an event object and truncates the result if necessary before printing it
    if isinstance(event, olympe.ArsdkMessageEvent):
      max_args_size = 200
      args = str(event.args)
      args = (args[: max_args_size - 3] + "...") if len(args) > max_args_size else args
      rospy.logdebug("{}({})".format(event.message.fullName, args))
    else:
      rospy.logdebug(str(event))


  # RC buttons listener     
  @olympe.listen_event(mapper.grab_button_event(_policy="wait")) 
  # https://developer.parrot.com/docs/olympe/arsdkng_mapper.html#olympe.messages.mapper.grab_button_event
  def on_grab_button_event(self, event, scheduler):
    # self.print_event(event)
    # button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
    # axis_button:	4 = max CCW yaw, 5 = max CW yaw, 6 = max trottle, 7 = min trottle
    # 		8 = min roll, 9 = max roll, 10 = min pitch, 11 = max pitch
    # 		12 = max camera down, 13 = max camera up, 14 = min zoom, 15 = max zoom
    if event.args["event"] == button_event.press:
      if event.args["button"] == 2: # left back button
        self.anafi._switch_to_manual_control()
        return
      if event.args["button"] == 3: # right back button
        self.anafi._switch_to_olympe_control()
        self.msg_rpyt = SkyControllerCommand()
        return


  # RC axis listener
  @olympe.listen_event(mapper.grab_axis_event(_policy="wait")) 
  # https://developer.parrot.com/docs/olympe/arsdkng_mapper.html#olympe.messages.mapper.grab_axis_event
  def on_grab_axis_event(self, event, scheduler):	
    # axis: 	0 = yaw, 1 = z, 2 = y, 3 = x, 4 = camera, 5 = zoom
    if event.args["axis"] == 0: # yaw
      self.msg_rpyt.yaw = event.args["value"]
    if event.args["axis"] == 1: # z
      self.msg_rpyt.z = event.args["value"]
    if event.args["axis"] == 2: # y/pitch
      self.msg_rpyt.y = event.argtrue
    self.msg_rpyt.header.stamp = rospy.Time.now()
    self.msg_rpyt.header.frame_id = '/body'
    self.anafi.pub_skycontroller.publish(self.msg_rpyt)
              

  # All other events
  @olympe.listen_event()
  def default(self, event, scheduler):
    pass


if __name__ == '__main__':
  rospy.init_node('anafi_bridge', anonymous = False)
  rospy.loginfo("AnafiBridge is running...")
  anafi = Anafi()	
  try:
    anafi.run()
  except rospy.ROSInterruptException:
    traceback.print_exc()
    pass
