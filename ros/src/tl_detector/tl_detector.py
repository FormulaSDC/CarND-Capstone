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

STATE_COUNT_THRESHOLD = 3
SAVE_FRAMES = False


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.car_current_waypoint = None
        self.lights = []
        self.stop_line_indices = []
        self.mode = 1  # 0 for development where we get the true light state
                       # 1 for testing where we predict the light state from a classifier

        cascade_name = 'cComb16x32LBPw30d2_3.xml'
        self.cascade = cv2.CascadeClassifier(cascade_name)

        # Subscribe
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/image_color', Image, self.image_cb)
        #to read raw images from rosbags
        sub3_1 = rospy.Subscriber('/image_raw', Image, self.image_cb)
        sub4 = rospy.Subscriber('/current_waypoint', Int32, self.current_waypoint_cb)

        if self.mode == 0:
            sub5 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                                self.traffic_cb)  # Color not available in real use.
                                

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    # Callback for current_pose topic
    def pose_cb(self, msg):
        self.pose = msg

    # Callback for current waypoint
    def current_waypoint_cb(self, waypoint_idx):
        self.car_current_waypoint = waypoint_idx.data
        # rospy.loginfo("current_waypoint_cb says car current waypoint is %s", self.car_current_waypoint)

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
        # rospy.loginfo("Getting stop_line_indices!!!")
        stop_line_positions = self.config['stop_line_positions']

        for stop_line_position in stop_line_positions:
            line_pose = Pose()
            line_pose.position.x = stop_line_position[0]
            line_pose.position.y = stop_line_position[1]

            # Find the nearest waypoint
            closest_wp = self.get_closest_waypoint(line_pose)
            # Add the stop line waypoint index to the list
            self.stop_line_indices.append(closest_wp)
            # rospy.loginfo("Line at position (%s,%s) and is closest to waypoint %s",
            #              line_pose.position.x, line_pose.position.y, closest_wp)

        # Output the list of indices
        # rospy.loginfo("Stop line waypoints are: %s ", self.stop_line_indices)

    def traffic_cb(self, msg):
        self.lights = msg.lights
        # Create a list with the status of each light -- corresponding to the list of waypoints
        # closest to each stop line and light
        self.stop_light_states = [light.state for light in self.lights]
        # Output the list of states
        # rospy.loginfo("Light states are: %s ", self.stop_light_states)

    # Callback for /image_color topic
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        # rospy.loginfo("image_cb starting!!!")

        # Get the image
        self.has_image = True
        self.camera_image = msg

        # Get the waypoint index of the next light (light_wp) and its state
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
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

            if (self.has_image):
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                detections = self.detect(cv_image)
            
                # detect TLs in the image returns array of detected TLs
                for tl_image in detections:
                    # TODO: determine active color
                    tl_color = self.light_classifier.get_classification(tl_image)
            else:
                self.prev_light_loc = None

            return TrafficLight.UNKNOWN

            

    # returns array of detected TLs
    def detect(self, image):
        img = cv2.resize(image, (720, 540));
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
        
        detection_res = self.cascade.detectMultiScale2(gray, 1.2, 1, 0,
                                             (16, 32),
                                             (100, 200));
        #return 0;
        detected = detection_res[0];
        images = [];
        img_out = img.copy();
        for result in detected:
            p0 = (result[0], result[1]);
            p1 = (p0[0] + result[2], p0[1] + result[3]);
            cv2.rectangle(img_out, p0, p1, (0, 0, 255), 2)
            images.append(cv2.resize(img[p0[1]:p1[1], p0[0]:p1[0], :],
                                     (16, 32)))
        cv2.imshow("detected", img_out);  cv2.waitKey(2);
        return images

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light_idx = None
        light_wp = -1

        # find the closest light
        if self.pose and (self.car_current_waypoint is not None):
            idx_dist_min = 9999
            for i, stop_line_idx in enumerate(self.stop_line_indices):
                idx_dist = stop_line_idx - self.car_current_waypoint
                if 0 < idx_dist < idx_dist_min:
                    light_idx = i
                    idx_dist_min = idx_dist
                    light_wp = stop_line_idx

        light_state = self.get_light_state(light_idx)
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