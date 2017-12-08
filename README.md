### Notes for the 20171208 commit to 'detection' branch
1. Traffic Light (TL) detection is enabled only if distance to the next 
TL is < 200 (or if we are reading images from the rosbag)

2. image_raw handler moved to the separate callback (to enable TL detection)
3. simple DNN (scripts/MixNet.py) for TL classification (accuracy 0.95 on val. set)
4. colored rectangles for detected TLs 
5. I've suppressed a couple of log messages 
in twist_controller\twist_controller.py
and waypoint_updater\waypoint_updater.py
with if (False):

I've tested on old simulator: it works, but it stops not on every red light, and
it always stops after 6600 waitpoints (even in mode 0, without detection).
On rosbags, it detects TL almost always.



### Notes for the 'detection' branch

I've modified tl_detector.py only_
1. line 40 - subscription for the '/image_raw' topic to use rosbag recordings
(with the same callback)

2. get_light_state in mode 1 only detects lights, it calls get_classification(),
bu6t always returns TrafficLight.UNKNOWN


3. detect(lines 216-234): I reduce image size to 720x540 and use rather large scale step (1.2).
I would prefer larger image and smaller step - but even these settings signioficantly affect vehicle dynamics.
**Is it possible to decouple detection from the control???**


4. process_traffic_lights now works in both modes to get light_wp.
How should we get light_wp on real car?

5. Collected images are at the https://www.dropbox.com/s/70kd715cuidi3hx/tl-detected.zip




### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
