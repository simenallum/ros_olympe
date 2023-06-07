# ROS Bridge for Parrot Drones
This ROS package contains interface to Olympe SDK version 7.5
Supports Parrot Anafi FPV drones

## Overview

**Original author:** Andriy Sarabakha, andriy.sarabakha@ntu.edu.sg<br />
**Updated by:** Simen Stensrød Allum and Øystein Solbø

## Installation

This package has been developed with **python3** in **ROS Noetic** on **Ubuntu 20.04**.

### Dependencies

- python3 -m pip install requirements.txt


## Run

    roslaunch ros_olympe <lab/sim>.launch use_manual_control:=<true/false> qualisys_available:=<true/false>
