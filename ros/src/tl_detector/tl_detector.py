#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np

MIN_DIST_TL = 200.  # Traffic lights farther than this distance are ignored
STATE_COUNT_THRESHOLD = 3  # require at least these many detections
DISPLAY_DETS = False  # Display detections on frames for debug
SAVE_FRAMES = False
TL_STATES = ['RED', 'YELLOW', 'GREEN', 'UNKNOWN', 'UNKNOWN']


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.car_current_waypoint = None
        self.lights = []
        self.stop_line_indices = []
        self.stop_light_states = None
        self.has_image = False

        self.mode = 1
        # 0 for development where we get the true traffic light state
        # 1 for testing where we predict the traffic light state from a classifier

        self.state = TrafficLight.UNKNOWN
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.unknown_count = 0

        self.cascade_name = 'cNewBag16x32LBPw30d2_3.xml'
        self.cascade = cv2.CascadeClassifier(self.cascade_name)

        # Subscribe
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/image_color', Image, self.image_cb)

        # To read raw images from rosbags :
        # Note : This is not required as we read images from /image_color
        # sub3_1 = rospy.Subscriber('/image_raw', Image, self.rosbag_cb)
        self.raw_image = False
        
        sub4 = rospy.Subscriber('/current_waypoint', Int32, self.current_waypoint_cb)

        if self.mode == 0:
            # Color not available in real use. Only for development mode
            sub5 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.spin()

    # Callback for current_pose topic
    def pose_cb(self, msg):
        self.pose = msg

    # Callback for current waypoint
    def current_waypoint_cb(self, waypoint_idx):
        self.car_current_waypoint = waypoint_idx.data
        # rospy.logdebug("current_waypoint_cb says car current waypoint is %s", self.car_current_waypoint)

    # Callback for /base_waypoints topic
    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

        # We get the waypoint indices nearest to stop lines here.  There are separate
        # position coordinates provided for stop lights and lines...
        # The difference between the lines and lights is that the line is where you
        # should actually stop, whereas the light is where the light itself is (i.e.
        # in the middle of the intersection rather than leading up to it).  Finding
        # the positions only needs to be done once.  The light state (available only
        # in development) can be queried separately from finding the locations

        # Get waypoint indices for the waypoint nearest to each stop LINE
        # rospy.logdebug("Getting stop_line_indices!!!")
        stop_line_positions = self.config['stop_line_positions']

        for stop_line_position in stop_line_positions:
            line_pose = Pose()
            line_pose.position.x = stop_line_position[0]
            line_pose.position.y = stop_line_position[1]

            # Find the nearest waypoint
            closest_wp = self.get_closest_waypoint(line_pose)
            # Add the stop line waypoint index to the list
            self.stop_line_indices.append(closest_wp)
            # rospy.logdebug("Line at position (%s,%s) and is closest to waypoint %s",
            #                line_pose.position.x, line_pose.position.y, closest_wp)

        # Output the list of indices
        # rospy.logdebug("Stop line waypoints are: %s ", self.stop_line_indices)

    def traffic_cb(self, msg):
        """Creates a list with the status of each light corresponding to the list
           of waypoints closest to each stop line and light"""
        self.lights = msg.lights
        self.stop_light_states = [light.state for light in self.lights]
        # Output the list of states
        # rospy.logdebug("Light states are: %s ", self.stop_light_states)

    # Callback for /image_raw topic
    def rosbag_cb(self, msg):
        # Get the image
        self.has_image = True
        self.camera_image = msg
        self.raw_image = True
        # Get the waypoint index of the next light (light_wp) and its state
        light_wp, state = self.process_traffic_lights()

    # Callback for /image_color topic
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        # rospy.logdebug("image_cb starting!!!")

        # Get the image
        self.has_image = True
        self.camera_image = msg
        self.raw_image = False
        
        # Get the waypoint index of the next light (light_wp) and its state
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if state == TrafficLight.UNKNOWN:
            self.unknown_count = self.unknown_count + 1
            if (self.unknown_count <= STATE_COUNT_THRESHOLD):
                return
        else:
            self.unknown_count = 0
        
        # treat yellow as red for stopping purposes
        if state == TrafficLight.YELLOW:
            state = TrafficLight.RED

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            if state not in [TrafficLight.RED, TrafficLight.YELLOW]:
                light_wp = -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    # Find the nearest waypoint
    def get_closest_waypoint(self, pose, waypoints=None):
        """Identifies the closest path waypoint to the given position
                https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
            waypoints : Default uses base waypoints
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        # Get the current x, y positions
        x = pose.position.x
        y = pose.position.y

        if waypoints is None:
            waypoints = self.waypoints

        # Find the nearest waypoint
        nearest_dist = 9999
        nearest_idx = -1
        for i in range(len(waypoints)):
            base_x = waypoints[i].pose.pose.position.x
            base_y = waypoints[i].pose.pose.position.y
            dist = self.dist(x, base_x, y, base_y)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = i
        return nearest_idx

    # Identify color of the specified (by ID) traffic light
    def get_light_state(self, light_idx=None):
        """Determines the current color of the traffic light
        Args:
            light_idx : index of light to classify if in development mode
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        # In development mode, we can get the ground truth light state
        if self.mode == 0:
            if light_idx is None:
                return TrafficLight.UNKNOWN
            else:
                return self.lights[light_idx].state
        else:
            if self.has_image:
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
                color = self.detect(cv_image)
                return color
            else:
                if not self.has_image:
                    # self.prev_light_loc = None
                    return TrafficLight.UNKNOWN

    def detect(self, image):
        """Detects traffic lights
        Args:
            image: image in BGR format
        Returns:
            color of the traffic light
        """
        img = cv2.resize(image, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detected = \
            self.cascade.detectMultiScale(gray, 1.2, 1, 0, (16, 32), (100, 200))


        img_out = img.copy() if DISPLAY_DETS else None
        colors_hist = np.zeros(TrafficLight.UNKNOWN+1, dtype=int)

        for (x,y,w,h) in detected:
            p0 = (x, y)
            p1 = (x+w, y+h)
            tl_image = cv2.resize(img[p0[1]:p1[1], p0[0]:p1[0], :], (16, 32))
            tl_color = self.light_classifier.get_classification(tl_image)
            if DISPLAY_DETS:
                color = (250, 250, 250)
                if tl_color == TrafficLight.RED:
                    color = (200, 0, 0)
                elif tl_color == TrafficLight.YELLOW:
                    color = (200, 200, 0)
                elif tl_color == TrafficLight.GREEN:
                    color = (0, 200, 0)
                cv2.rectangle(img_out, p0, p1, color, 2)
            colors_hist[tl_color] = colors_hist[tl_color] + 1

        # special tweak to:
        #  - return TrafficLight.UNKNOWN if there were no 'real' lights
        #  - if any KNOWN light was detected, return it ()
        colors_hist[TrafficLight.UNKNOWN] = 1
        result = np.argmax(colors_hist)

        # display detection results
        if DISPLAY_DETS:
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            img_out = cv2.resize(img_out, None, fx=.5, fy=.5)
            cv2.imshow("detected", img_out)
            cv2.waitKey(1)

        return result

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light_idx = None
        light_wp = -1
        light_state = TrafficLight.UNKNOWN

        # find the closest light
        idx_dist_min = 9999
        if self.pose and (self.car_current_waypoint is not None):
            for i, stop_line_idx in enumerate(self.stop_line_indices):
                idx_dist = stop_line_idx - self.car_current_waypoint
                if 0 < idx_dist < idx_dist_min:
                    light_idx = i
                    idx_dist_min = idx_dist
                    light_wp = stop_line_idx
        if self.raw_image or idx_dist_min < MIN_DIST_TL:
            light_state = self.get_light_state(light_idx)
        
        rospy.logdebug("light_state = %s", TL_STATES[light_state])
        return light_wp, light_state

    # Euclidean distance.
    def dist(self, x1, x2, y1, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if __name__ == '__main__':
    try:
        cnt = 0
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
