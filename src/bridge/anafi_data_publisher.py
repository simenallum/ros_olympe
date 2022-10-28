#!/usr/bin/python3

import rospy
import cv2
import queue
import threading
import traceback
import math
import olympe
import numpy as np
import pymap3d
import os

from std_msgs.msg import UInt8, UInt16, Int8, String, Header, Time
from geometry_msgs.msg import PoseStamped, QuaternionStamped, TwistStamped, Vector3Stamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, NavSatFix, NavSatStatus

from olympe.messages.ardrone3.Piloting import Emergency
from olympe.messages.ardrone3.PilotingState import SpeedChanged, AttitudeChanged
from olympe.messages.ardrone3.PilotingSettingsState import MaxTiltChanged, MaxDistanceChanged, MaxAltitudeChanged, NoFlyOverMaxDistanceChanged, BankedTurnChanged
from olympe.messages.ardrone3.SpeedSettingsState import MaxVerticalSpeedChanged, MaxRotationSpeedChanged, MaxPitchRollRotationSpeedChanged

from olympe.messages import mapper
from olympe.enums.mapper import button_event

from olympe_bridge.msg import SkyControllerCommand, Float32Stamped
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R


olympe.log.update_config({"loggers": {"olympe": {"level": "ERROR"}}})
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT", "554")


class AnafiBridgeDataPublisher:
  def __init__(
        self, 
        anafi, 
        config    
      ) -> None:

    # Initializing reference to connected drone
    self.anafi = anafi

    # Initializing parameters
    self.config = config
    self.cv_bridge = CvBridge()

    # Initializing video streaming
    self._initialize_video_stream()

    # Initializing publishers
    self.pub_image = rospy.Publisher("/anafi/image", Image, queue_size=1)
    self.pub_time = rospy.Publisher("/anafi/time", Time, queue_size=1)
    self.pub_attitude = rospy.Publisher("/anafi/attitude", QuaternionStamped, queue_size=1)
    self.pub_gnss_location = rospy.Publisher("/anafi/gnss_location", NavSatFix, queue_size=1)
    self.pub_height = rospy.Publisher("/anafi/height", Float32Stamped, queue_size=1)
    self.pub_optical_flow_velocities = rospy.Publisher("/anafi/optical_flow_velocities", Vector3Stamped, queue_size=1)
    self.pub_link_goodput = rospy.Publisher("/anafi/link_goodput", UInt16, queue_size=1)
    self.pub_link_quality = rospy.Publisher("/anafi/link_quality", UInt8, queue_size=1)
    self.pub_wifi_rssi = rospy.Publisher("/anafi/wifi_rssi", Int8, queue_size=1)
    self.pub_battery = rospy.Publisher("/anafi/battery", UInt8, queue_size=1)
    self.pub_state = rospy.Publisher("/anafi/state", String, queue_size=1)
    self.pub_mode = rospy.Publisher("/anafi/mode", String, queue_size=1)
    self.pub_pose = rospy.Publisher("/anafi/pose", PoseStamped, queue_size=1)
    self.pub_odometry = rospy.Publisher("/anafi/odometry", Odometry, queue_size=1)
    self.pub_rpy = rospy.Publisher("/anafi/rpy", Vector3Stamped, queue_size=1)
    self.pub_skycontroller = rospy.Publisher("/skycontroller/command", SkyControllerCommand, queue_size=1)
    self.pub_polled_velocities = rospy.Publisher("/anafi/polled_body_velocities", TwistStamped, queue_size=1)
    self._initialize_qualisys_cb()

    # Initialize the eventlistener
    self.every_event_listener = EveryEventListener(self.anafi, self)
    self.every_event_listener.subscribe()


  def _initialize_video_stream(self) -> None:
    self.frame_queue = queue.Queue()
    self.flush_queue_lock = threading.Lock()

    if DRONE_RTSP_PORT is not None:
      self.anafi.drone.streaming.server_addr = self.config.drone_ip + f":{DRONE_RTSP_PORT}"

    # Setup the callback functions to do some live video processing
    self.anafi.drone.streaming.set_callbacks(
      raw_cb=self._yuv_frame_cb,
      flush_raw_cb=self._flush_cb
    )

    self.anafi.drone.streaming.start()


  def _yuv_frame_cb(self, yuv_frame) -> None:  
    # with self.flush_queue_lock: # Uncertain whether the lock is necessary
    yuv_frame.ref()
    self.frame_queue.put_nowait(yuv_frame)


  def _flush_cb(self, _) -> bool:
    with self.flush_queue_lock:
      while not self.frame_queue.empty():
        self.frame_queue.get_nowait().unref()
    return True


  def _initialize_qualisys_cb(self) -> None:
    if self.config.is_qualisys_available:			
      rospy.loginfo("Flying at drone lab: Qualisys is available")
    else:
      rospy.loginfo("Not flying at the lab: Qualisys is unavailable")

    if self.config.is_qualisys_available:
      rospy.Subscriber("/qualisys/Anafi/pose", PoseStamped, self._qualisys_callback)
      self.last_received_location = NavSatFix()


  def _qualisys_callback(self, msg: PoseStamped):
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z

    ell_wgs84 = pymap3d.Ellipsoid('wgs84')
    lat0, lon0, h0 = 63.418215, 10.401655, 0   # Origin of ENU, setting origin to be at the drone lab at NTNU

    lat1, lon1, h1 = pymap3d.enu2geodetic(x, y, z, \
                      lat0, lon0, h0, \
                      ell=ell_wgs84, deg=True)  # wgs84 ellisoid

    self.last_received_location.latitude = lat1
    self.last_received_location.longitude = lon1
    self.last_received_location.altitude = h1


  def _yuv_callback(self, yuv_frame):
    # Use OpenCV to convert the yuv frame to RGB
    # the VideoFrame.info() dictionary contains some useful information
    # such as the video resolution
    info = yuv_frame.info()
    height, width = (  
      info["raw"]["frame"]["info"]["height"],
      info["raw"]["frame"]["info"]["width"],
    )

    # Convert pdraw YUV flag to OpenCV YUV flag
    cv2_cvt_color_flag = {
      olympe.VDEF_I420 : cv2.COLOR_YUV2BGR_I420,
      olympe.VDEF_NV12 : cv2.COLOR_YUV2BGR_NV12,
    }[yuv_frame.format()]

    # Use OpenCV to convert the yuv frame to RGB
    cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag) 
    
    # Convert from OpenCV-image to ROS-image
    msg_image = self.cv_bridge.cv2_to_imgmsg(cv2frame, "bgr8")
    self.pub_image.publish(msg_image)

    # Extract metadata included in the drone's videostream 
    metadata = yuv_frame.vmeta()

    if metadata[1] != None:
      header = Header()
      header.stamp = rospy.Time.now()
      header.frame_id = '/body'
    
      frame_timestamp = info['raw']['frame']['timestamp'] # timestamp [microsec]
      secs = int(frame_timestamp // 1e6)
      microsecs = (frame_timestamp - (1e6 * secs))
      nanosecs = microsecs * 1e3
      msg_time = Time()
      msg_time.data = rospy.Time(secs, nanosecs) 
      self.pub_time.publish(msg_time)

      if "drone" in metadata[1]:
        drone_quat = metadata[1]['drone']['quat'] # attitude
        quat = [drone_quat['x'], drone_quat['y'], drone_quat['z'], drone_quat['w']]

        Rot = R.from_quat(quat)
        drone_rpy = Rot.as_euler('xyz', degrees=False)

        # TODO Rewrite this part to more readable code. Currenly using magic number etc.
        # This part adds the offsets to the quaternions before it is published
        if self.config.drone_ip == "192.168.53.1":
          # Real drone
          correction_terms = (-0.015576460455065291, -0.012294590577876349, 0) #(-0.009875596168668191, -0.006219417359313843, 0)
        else:
          # Simulator
          correction_terms = (-0.0002661987518324087, -0.0041069024624204534, 0)

        drone_rpy_corrected = drone_rpy + correction_terms
        rot_corrected = R.from_euler('xyz', drone_rpy_corrected, degrees=False)
        quat = R.as_quat(rot_corrected)

        msg_attitude = QuaternionStamped()
        msg_attitude.header = header
        msg_attitude.quaternion = Quaternion(quat[0], quat[1], quat[2], quat[3])
        self.pub_attitude.publish(msg_attitude)

        msg_location = NavSatFix()
        status = NavSatStatus()
        if "location" in metadata[1]['drone']:
          location = metadata[1]['drone']['location']       
          if location != {}:			
            msg_location.header = header
            msg_location.header.frame_id = '/world'
            msg_location.latitude = location['latitude']    # [deg]
            msg_location.longitude = location['longitude']  # [deg]
            msg_location.altitude = location['altitude']    # [m] over WGS84

            status.status = 0 	# Position fix

          else:
            # No connection - setting to zero to make it explicit
            msg_location.latitude = 0   # [deg]
            msg_location.longitude = 0  # [deg]
            msg_location.altitude = 0   # [m] over WGS84

            status.status = -1	# No position fix

        elif self.config.is_qualisys_available:
          msg_location.header = header
          msg_location.header.frame_id = '/world'
          msg_location.latitude = self.last_received_location.latitude
          msg_location.longitude = self.last_received_location.longitude
          msg_location.altitude = self.last_received_location.altitude

          status.status = 0	# Position fix

        msg_location.status = status
        self.pub_gnss_location.publish(msg_location)

        ground_distance = metadata[1]['drone']['ground_distance'] # barometer (m)
        height_msg = Float32Stamped()
        height_msg.header = header
        height_msg.data = ground_distance
        self.pub_height.publish(height_msg)

        speed = metadata[1]['drone']['speed'] # opticalflow speed (m/s)
        msg_speed = Vector3Stamped()
        msg_speed.header = header
        msg_speed.header.frame_id = '/world'
        msg_speed.vector.x = speed['north']
        msg_speed.vector.y = speed['east']
        msg_speed.vector.z = speed['down']
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
        msg_odometry.child_frame_id = '/body'
        msg_odometry.pose.pose = msg_pose.pose
        msg_odometry.twist.twist.linear.x =  math.cos(drone_rpy[2])*msg_speed.vector.x + math.sin(drone_rpy[2])*msg_speed.vector.y
        msg_odometry.twist.twist.linear.y = -math.sin(drone_rpy[2])*msg_speed.vector.x + math.cos(drone_rpy[2])*msg_speed.vector.y
        msg_odometry.twist.twist.linear.z = msg_speed.vector.z
        self.pub_odometry.publish(msg_odometry)

        battery_percentage = metadata[1]['drone']['battery_percentage'] # [0=empty, 100=full]
        self.pub_battery.publish(battery_percentage)

        state = metadata[1]['drone']['flying_state'] # ['LANDED', 'MOTOR_RAMPING', 'TAKINGOFF', 'HOVERING', 'FLYING', 'LANDING', 'EMERGENCY']
        self.pub_state.publish(state)

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
        link_goodput = metadata[1]['links'][0]['wifi']['goodput'] # throughput of the connection (b/s)
        self.pub_link_goodput.publish(link_goodput)

        link_quality = metadata[1]['links'][0]['wifi']['quality'] # [0=bad, 5=good]
        self.pub_link_quality.publish(link_quality)

        wifi_rssi = metadata[1]['links'][0]['wifi']['rssi'] # signal strength [-100=bad, 0=good] (dBm)
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


  def run(self) -> None:
    rospy.logdebug('MaxTilt = %f [%f, %f]', self.anafi.drone.get_state(MaxTiltChanged)["current"], self.anafi.drone.get_state(MaxTiltChanged)["min"], self.anafi.drone.get_state(MaxTiltChanged)["max"])
    rospy.logdebug('MaxVerticalSpeed = %f [%f, %f]', self.anafi.drone.get_state(MaxVerticalSpeedChanged)["current"], self.anafi.drone.get_state(MaxVerticalSpeedChanged)["min"], self.anafi.drone.get_state(MaxVerticalSpeedChanged)["max"])
    rospy.logdebug('MaxRotationSpeed = %f [%f, %f]', self.anafi.drone.get_state(MaxRotationSpeedChanged)["current"], self.anafi.drone.get_state(MaxRotationSpeedChanged)["min"], self.anafi.drone.get_state(MaxRotationSpeedChanged)["max"])
    rospy.logdebug('MaxPitchRollRotationSpeed = %f [%f, %f]', self.anafi.drone.get_state(MaxPitchRollRotationSpeedChanged)["current"], self.anafi.drone.get_state(MaxPitchRollRotationSpeedChanged)["min"], self.anafi.drone.get_state(MaxPitchRollRotationSpeedChanged)["max"])
    rospy.logdebug('MaxDistance = %f [%f, %f]', self.anafi.drone.get_state(MaxDistanceChanged)["current"], self.anafi.drone.get_state(MaxDistanceChanged)["min"], self.anafi.drone.get_state(MaxDistanceChanged)["max"])
    rospy.logdebug('MaxAltitude = %f [%f, %f]', self.anafi.drone.get_state(MaxAltitudeChanged)["current"], self.anafi.drone.get_state(MaxAltitudeChanged)["min"], self.anafi.drone.get_state(MaxAltitudeChanged)["max"])
    rospy.logdebug('NoFlyOverMaxDistance = %i', self.anafi.drone.get_state(NoFlyOverMaxDistanceChanged)["shouldNotFlyOver"])
    rospy.logdebug('BankedTurn = %i', self.anafi.drone.get_state(BankedTurnChanged)["state"])
    
    i = 0
    min_iterations = 9
    prev_attitude = None

    while not rospy.is_shutdown():
      connection = self.anafi.drone.connection_state()
      if not connection:
        # Discontinue if lost connection
        rospy.logfatal("Connection lost")
        self._flush_cb(None)
        break

      if i >= min_iterations:
        att_euler = self.anafi.drone.get_state(AttitudeChanged)

        roll = att_euler["roll"]     
        pitch = att_euler["pitch"]    
        yaw = att_euler["yaw"]  

        # Check if new 5 Hz data has arrived
        if prev_attitude is None or prev_attitude != [roll, pitch, yaw]:
          prev_attitude = [roll, pitch, yaw]

          # Check if new telemetry
          pos_dot_ned = self.anafi.drone.get_state(SpeedChanged)
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

            return np.array(
              [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
              ]
            )

          def Ry(radians):
            c = np.cos(radians)
            s = np.sin(radians)

            return np.array(
              [
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
              ]
            )

          def Rz(radians):
            c = np.cos(radians)
            s = np.sin(radians)

            return np.array(
              [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
              ]
            )

          # Tried to use scipy's rotation-matrix, but wouldn't get quite right...
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
                
      rospy.sleep(0.009)


class EveryEventListener(olympe.EventListener):
  def __init__(self, anafi, anafi_bridge_publisher) -> None:
    super().__init__(anafi.drone)
    self.anafi = anafi
    self.anafi_bridge_publisher = anafi_bridge_publisher
    self.msg_rpyt = SkyControllerCommand()


  def print_event(self, event) -> None: 
    # Serializes an event object and truncates the result if necessary before printing it
    if isinstance(event, olympe.ArsdkMessageEvent):
      max_args_size = 200
      args = str(event.args)
      args = (args[: max_args_size - 3] + "...") if len(args) > max_args_size else args
      rospy.logdebug("{}({})".format(event.message.fullName, args))
    else:
      rospy.logdebug(str(event))


  # RC buttons listener     
  @olympe.listen_event(mapper.grab_button_event()) 
  # https://developer.parrot.com/docs/olympe/arsdkng_mapper.html#olympe.messages.mapper.grab_button_event
  def on_grab_button_event(self, event, scheduler):
    self.print_event(event)
    # button: 	0 = RTL, 1 = takeoff/land, 2 = back left, 3 = back right
    # axis_button:	4 = max CCW yaw, 5 = max CW yaw, 6 = max trottle, 7 = min trottle
    # 		8 = min roll, 9 = max roll, 10 = min pitch, 11 = max pitch
    # 		12 = max camera down, 13 = max camera up, 14 = min zoom, 15 = max zoom
    if event.args["event"] == button_event.press:
      if event.args["button"] == 0: # RTL
        self.anafi.drone(Emergency()).wait()
        rospy.logfatal("Emergency!!!")
        return
      if event.args["button"] == 2: # left back button
        self.anafi.switch_manual()
        return
      if event.args["button"] == 3: # right back button
        self.anafi.switch_offboard()
        self.msg_rpyt = SkyControllerCommand()
        return


  # RC axis listener
  @olympe.listen_event(mapper.grab_axis_event()) 
  # https://developer.parrot.com/docs/olympe/arsdkng_mapper.html#olympe.messages.mapper.grab_axis_event
  def on_grab_axis_event(self, event, scheduler):	
    # axis: 	0 = yaw, 1 = z, 2 = y, 3 = x, 4 = camera, 5 = zoom
    if event.args["axis"] == 0: # yaw
      self.msg_rpyt.yaw = event.args["value"]
    if event.args["axis"] == 1: # z
      self.msg_rpyt.z = event.args["value"]
    if event.args["axis"] == 2: # y/pitch
      self.msg_rpyt.y = event.args["value"]
    if event.args["axis"] == 3: # x/roll
      self.msg_rpyt.x = event.args["value"]
    self.msg_rpyt.header.stamp = rospy.Time.now()
    self.msg_rpyt.header.frame_id = '/body'
    self.anafi_bridge_publisher.pub_skycontroller.publish(self.msg_rpyt)


  @olympe.listen_event(AttitudeChanged(_policy="wait"))
  def onAttitudeChanged(self, event, scheduler):
    msg_rpy = Vector3Stamped()
    msg_rpy.header.stamp = rospy.Time.now()
    msg_rpy.header.frame_id = '/world'
    msg_rpy.vector.x = event.args["roll"]
    msg_rpy.vector.y = event.args["pitch"]
    msg_rpy.vector.z = event.args["yaw"]
    self.anafi_bridge_publisher.pub_rpy.publish(msg_rpy)
              

  # All other events
  @olympe.listen_event()
  def default(self, event, scheduler):
    pass
