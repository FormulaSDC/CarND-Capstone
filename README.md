## Udacity Self-Driving Car Nanodegree
## Capstone Project

### Team FormulaSDC
| Name                       | Email                    |
|:---------------------------|:-------------------------|
| Prerit Jaiswal             | prerit.jaiswal@gmail.com |
| Anton Varfolomeev          | dizvara@gmail.com        |
| Kemal Tepe                 | ketepe@gmail.com         |
| Paul Walker                | n43tc3d2rp-u1@yahoo.com  |
| Matthias von dem Knesebeck | mail@knesebeck.com       |


### Overview
Following were the main objectives of this project : 

* Smoothly follow waypoints in the simulator. 
* Respect the target top speed. 
* Stop at traffic lights when needed.
* Stop and restart controllers when DBW is disables/enabled.
* Publish throttle, steering, and brake commands at 50 Hz.

To achieve these objectives, we implemented a finite state machine consisting of 3 states : (i) `go` state,  (ii) `stop` state, and (iii) `idle` state. In the absence of a traffic light or if the light is green, the state is set to `go` with target speed set to the speed limit while ensuring that the transition from current to target speed is smooth. If a red or yellow traffic light is detected, state is set to `stop` if it is possible to bring the car to halt without exceeding maximum braking. Again, a smooth transition is implemented from current speed to 0. Once the car has come to halt, state is changed to `idle`. The speed in `idle` state is set to zero and the car remains in this state until the light turns green and car goes back to `go` state.  For yaw control, we have used `YawController` already provided while for throttle, we used a proportional controller which takes as input the error in speed.   

### Traffic Light Detection
The traffic light detection has been realized by implementing a Cascade Classifier. Once a traffic light has been identified, the bounding box is scaled to a 16x32 pixel image. This image is then supplied to a color detection neural network that was trained with numerous examples from the labeled Bosch Traffic Light Dataset as well as samples from the Udacity simulation track. This network returns the color with the highest resulting probability identified. The Traffic Light Detector then publishes the traffic light waypoint once at least 3 consecutive images have been identified with the same color.

### Results 

Car was able to successfully complete track lap while meeting all the objectives. Here is a video demonstration on simulator track: 

[![Simulator](http://img.youtube.com/vi/9MybAoVeOkI/0.jpg)](http://www.youtube.com/watch?v=9MybAoVeOkI "Simulator")

The results are presented in the following image samples from the simulator track. The bounding boxes show the detected colors:

#### Detection Result for "Green" Traffic Light 
<img src="imgs/screenshot_green.png" width="300" >

#### Detection Result for "Yellow" Traffic Light 
<img src="imgs/screenshot_yellow.png" width="300">

#### Detection Result for "Red" Traffic Light 
<img src="imgs/screenshot_red.png" width="300">



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
