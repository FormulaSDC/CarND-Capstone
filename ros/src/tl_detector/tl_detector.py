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

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        # Initialize variables
        self.pose = None
        self.line_waypoints = None
        self.car_waypoints = None
        self.car_current_waypoint = -1
        self.camera_image = None
        self.lights = []

        # Initialize stop light indices
        self.got_light_indices = False
        self.stop_light_indices = []

        # Subscribe
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)  # Color not available in real use.  See note below
        sub4 = rospy.Subscriber('/image_color', Image, self.image_cb) # Was sub6... any reason???
        sub5 = rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb)

        '''
        NOTE REGARDING /vehicle/traffic_lights:
            This topic rovides you with the location of the traffic light in 3D map space and
            helps you acquire an accurate ground truth data source for the traffic light
            classifier by sending the current color state of all traffic lights in the
            simulator. When testing on the vehicle, the color state will not be available. You'll need to
            rely on the position of the light and the camera image to predict it.
        '''

        # Provides the locations of the stop lights
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Publishes the index of the waypoint closest to the stop line for the stop light
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # Initialize objects
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        # Initialize variables
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Process comments
        rospy.spin()

    # Callback for current_pose topic
    def pose_cb(self, msg):
        self.pose = msg

    # Callback for final_waypoints topic
    def final_waypoints_cb(self, waypoints):
        self.car_waypoints = waypoints
        self.car_current_waypoint = self.car_waypoints.waypoints[0].pose.header.seq
        rospy.loginfo("final_waypoints_cb says car current waypoint is %s",self.car_current_waypoint)

    # Callback for /base_waypoints topic
    def waypoints_cb(self, waypoints):
        self.line_waypoints = waypoints

        # We get the waypoint indices nearest to stop lines here.  There are separate
        # position coordinates provided for stop lights and lines...
        # The difference between the lines and lights is that the line is where you
        # should actually stop, whereas the light is where the light itself is (i.e.
        # in the middle of the intersection rather than leading up to it).  Finding
        # the positions only needs to be done once.  The light state (available only
        # in development) can be queried separately from finding the locations

        # Get waypoint indices for the waypoint nearest to each stop LINE
        rospy.loginfo("Getting stop_line_indices!!!")
        stop_line_positions = self.config['stop_line_positions']
        self.stop_line_indices = []
        nLines = len(stop_line_positions)
        for i in range(nLines):
          line_x = stop_line_positions[i][0]
          line_y = stop_line_positions[i][1]
          # Find the nearest waypoint
          for j in range(len(self.line_waypoints.waypoints)):
            base_x = self.line_waypoints.waypoints[j].pose.pose.position.x
            base_y = self.line_waypoints.waypoints[j].pose.pose.position.y
            dist = self.dist(line_x,base_x,line_y,base_y)
            if j==0:
              mindist=dist
              minpt=0
            else:
              if dist<mindist:
                mindist=dist
                minpt=j
          # Add the stop line waypoint index to the list
          self.stop_line_indices.append(minpt)
          rospy.loginfo("Line %s is at position (%s,%s) and is closest to waypoint %s", i, line_x, line_y, minpt)

        # Output the list of indices
        rospy.loginfo("Line indices are: %s ", self.stop_line_indices)

    # Callback for /vehicle/traffic_lights topic
    def traffic_cb(self, msg):
        self.lights = msg.lights

        # Create a list with the status of each light -- corresponding to the list of waypoints
        # closest to each stop line and light
        self.stop_light_states = []
        nLights = len(self.lights)
        for i in range(nLights):
          self.stop_light_states.append(self.lights[i].state)

        # Output the list of states
        rospy.loginfo("Light states are: %s ", self.stop_light_states)

        # Retrieve position and orientation
        #p_x = self.lights[i].pose.pose.position.x
        #p_y = self.lights[i].pose.pose.position.y
        #p_z = self.lights[i].pose.pose.position.z
        #o_x = self.lights[i].pose.pose.orientation.x
        #o_y = self.lights[i].pose.pose.orientation.y
        #o_z = self.lights[i].pose.pose.orientation.z
        #o_w = self.lights[i].pose.pose.orientation.w
        #rospy.loginfo("Light %s is %s at position (%s,%s,%s) and orientation (%s,%s,%s,%s)", i, color, p_x, p_y, p_z, o_x, o_y, o_z, o_w)

    # Callback for /image_color topic
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        rospy.loginfo("image_cb starting!!!")

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

    # Process traffic lights
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
           location and color.
           NOTE: I don't understand the "if one exists" condition here.  A next traffic
                 light always exists since the track is a loop.  Not sure how to determine
                 whether it is visible or not, maybe with the classifier later?
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        rospy.loginfo("process_traffic_lights starting!!!")
        #light = None

        # Find the closest visible traffic light (if one exists).  TODO: We assume for now that
        # waypoint indices are increasing.  Relax this assumption later!!!
        if self.car_current_waypoint >= 0:
          # Loop through the stop lines and get the next visible light
          for i in range(len(self.stop_line_indices)):
            dst = self.wpdist(self.stop_line_indices[i],self.car_current_waypoint)
            rospy.loginfo("distance to line=%s with waypoint index %s is %s", i, self.stop_line_indices[i],dst)
            # Initialize
            if i==0:
              nextline_dst = dst
              nextline_idx = i
              nextline_wp  = self.stop_line_indices[i]
            # Update if closer
            else:
              if (dst<nextline_dst):
                nextline_dst = dst
                nextline_idx = i
                nextline_wp  = self.stop_line_indices[i]

          # Echo back the results
          rospy.loginfo("The next stop line is at waypoint %s", nextline_wp)

          # Get the state of the corresponding light -- passing in 0 tells
          # get_light_state to use ground truth rather than the detector which
          # has not yet been built
          state = self.get_light_state(nextline_idx,0)

          # Convert the state to a color for printing
          if state == 2:   color='green'
          elif state == 1: color='yellow'
          elif state == 0: color='red'
          else:            color='unknown'

          # Echo back the results
          rospy.loginfo("The light is %s", color)

          # Finally, return the waypoint of the next light and its state
          return nextline_wp, state

        # If we don't know where the car is, then we also do not know where the next light is...
        # In this case, we return some invalid codes... -1 for the waypoint location, and 4 (unknonwn)
        # for the state.  Recall the state definitions are:
        #      UNKNOWN=4
        #      GREEN=2
        #      YELLOW=1
        #      RED=0
        else:
          return -1, 4

    # Get closest waypoint... not needed in my implementation because I get
    # this value from the /final_waypoint topic published by waypoint_updater.
    # Not really clear why it was not designed this way from the start.
    #def get_closest_waypoint(self, pose):
    #    """Identifies the closest path waypoint to the given position
    #        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
    #    Args:
    #        pose (Pose): position to match a waypoint to
    #
    #    Returns:
    #        int: index of the closest waypoint in self.waypoints
    #
    #    """
    #    #TODO implement
    #    return 0

    # Identify color of the specified (by ID) trafic light
    def get_light_state(self, light_idx, mode):
        """Determines the current color of the traffic light

        Args:
            light_idx (TrafficLight): index of light to classify
            mode: 0 for development where we get the true light state
                  1 for testing where we predict the light state from a classifier

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # In development mode, we can get the ground truth light state
        if mode==0:
          return self.lights[light_idx].state

        # In testing mode, we predict the light state using a classifier
        else:
            if(not self.has_image):
                self.prev_light_loc = None
                return False

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            #Get classification
            return self.light_classifier.get_classification(cv_image)

    # Euclidean distance.  TODO: Find a common place to put this function
    def dist( self, x1, x2, y1, y2 ):
      dist = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
      return dist

    # "waypoint distance" function.  Calculates the number of waypoints between
    # two waypoint indices.  This is only needed because the waypoints loop back
    # around and restart the index.
    def wpdist(self, light_wp, car_wp):
      # Get the total number of waypoints on the track
      n = len(self.line_waypoints.waypoints)
      # Handle threee cases
      if light_wp == car_wp: wpdist=0
      elif light_wp > car_wp: wpdist = light_wp - car_wp
      else: wpdist = (n-car_wp) + light_wp + 1
      return wpdist

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
