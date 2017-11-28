#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tf
from itertools import cycle, islice, dropwhile

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):

        rospy.init_node('waypoint_updater')
        # rospy.loginfo('WaypointUpdater: __init__ starting')

        # Initialize variable
        self._base_waypoints = None
        self.n_base_wps = None
        self._current_pose = None
        self.red_light_wp = -1
        self.current_linear_velocity = -1
        self.car_state = "go" # Possible states will be: go, stop, idle

        # Current position:
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        # Current velocity
        #rospy.Subscriber('/current_velocity', TwistStamped , self.current_velocity_cb)

        # Base waypoints: are for the entire track and are only published once
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Final waypoints: First waypoint listed is the one directly in front of the car.
        #                  Total number of way points to include are given above by LOOKAHEAD_WPS
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        # rospy.spin()
        self.loop()


    def loop(self):
        rate = rospy.Rate(20) # 25 Hz
        while not rospy.is_shutdown():
            if (self._base_waypoints is None) or (self._current_pose is None):
                continue

            # Find the nearest waypoint
            self._nearest = self.find_nearest(self._current_pose, self._base_waypoints)
            rospy.loginfo("nearest waypoint index = %s", self._nearest)

            # Create the lane object to be published as final_waypoints
            myLane = Lane()

            # Create its header
            myLane.header.seq = 0
            myLane.header.stamp = rospy.Time(0)
            myLane.header.frame_id = '/world'

            # Create the waypoints locations
            myLane.waypoints = []
            index = self._nearest
            #rospy.loginfo("nearest waypoint index : %s of %s", index, self.n_base_wps)
            last_wp = index + LOOKAHEAD_WPS - 1  # Last waypoint

            if index + LOOKAHEAD_WPS > self.n_base_wps:
                # https://stackoverflow.com/questions/8940737/cycle-through-list-starting-at-a-certain-element
                cycled = cycle(self._base_waypoints)
                sliced = islice(cycled, index, last_wp+1)
                base_wps = list(sliced)
            else:
                base_wps = self._base_waypoints[index: last_wp+1]

            for i, base_wp in enumerate(base_wps):
                # Copy in the relevant waypoint
                myLane.waypoints.append(base_wp)

                # But modify the sequence number to contain the index number... will use later in tl_detector.py
                myLane.waypoints[i].pose.header.seq = (index + i) % self.n_base_wps

            # last_wps = (self._nearest + LOOKAHEAD_WPS - 1) % nWp

            # Set velocities based on state...
            # rospy.loginfo("car_state= %s", self.car_state)

            # # Set velocity in the "go" state -- other states will be added later
            # if self.car_state == "go":
            #
            #     # Accelerate to target velocity -- right now this is just a constant, but once the DBW module
            #     # is ready, this will accelerate gradually
            #
            #     for i in range(LOOKAHEAD_WPS):
            #         self.set_waypoint_velocity(myLane.waypoints, i, new_velocity)

            # Finally, publish it
            self.final_waypoints_pub.publish(myLane)

            # Wait a little before publishing the next command
            rate.sleep()

    def pose_cb(self, msg):
        # rospy.loginfo('WaypointUpdater: pose_cb starting')

        # Get the current position
        self._current_pose = msg.pose
        # rospy.loginfo("_current_pose: %s", self._current_pose)

    # Gets the waypoints
    def waypoints_cb(self, waypoints):
        #r ospy.loginfo('WaypointUpdater: waypoints_cb starting')
        self._base_waypoints = waypoints.waypoints
        self.n_base_wps = len(self._base_waypoints)

    # Callback function for current_velocity
    def current_velocity_cb(self, msg):
        self.current_linear_velocity = msg.twist.linear.x  # m/s

    # Find the nearest waypoint
    def find_nearest(self, curr_pose, base_wps):
        # Get the current x, y positions
        curr_x = curr_pose.position.x
        curr_y = curr_pose.position.y
        curr_yaw = self.get_yaw(curr_pose.orientation)
        # rospy.loginfo("current position (x,y) = (%s,%s)" , curr_x, curr_y)

        # Find the nearest waypoint
        nearest_dist = 9999
        nearest_idx = -1
        for i in range(self.n_base_wps):
            base_x = base_wps[i].pose.pose.position.x
            base_y = base_wps[i].pose.pose.position.y
            dist = self.dist(curr_x, base_x, curr_y, base_y)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = i

        nearest_wp_x = base_wps[nearest_idx].pose.pose.position.x
        nearest_wp_y = base_wps[nearest_idx].pose.pose.position.y
        heading = math.atan2(nearest_wp_y - curr_y, nearest_wp_x - curr_x)
        if abs(heading - curr_yaw) > math.pi/4:
            nearest_idx += 1
        return nearest_idx % self.n_base_wps

    def get_yaw(self, orientation):
        quaternion = (orientation.x,
                      orientation.y,
                      orientation.z,
                      orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]

    # Euclidean distance.  TODO: Find a common place to put this function
    def dist(self, x1, x2, y1, y2):
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return dist

    def traffic_cb(self, msg):
        self.red_light_wp = msg.data
        # rospy.loginfo("waypoint_updater:traffic_cb says there is a
        # red light at waypoint %s" , self.red_light_wp )

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
