std_msgs/Header header

float64 latitude	        # [deg]
float64 longitude	        # [deg]
float64 altitude	        # [m]
float32 heading 	        # heading w.r.t. North [deg] TODO: Check if this is in rad, as was the case of movebycommand
uint32 orientation_mode 	# orientation mode {TO_TARGET, HEADING_START, HEADING_DURING}

# Orientation-mode:
# NONE: The drone won’t change its orientation (0)
# TO_TARGET: The drone will make a rotation to look in direction of the given location (1)
# HEADING_START: The drone will orientate itself to the given heading before moving to the location (2)
# HEADING_DURING: The drone will orientate itself to the given heading while moving to the location (3)
