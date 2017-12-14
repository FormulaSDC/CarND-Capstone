#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tf
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
EASY_DECEL = 1.  # used to determine the stopping distance
HARD_DECEL = 5.  # abort stopping if deceleration required exceeds this threshold
ONE_MPH = 0.44704
SAFE_DIST = 8.

class WaypointUpdater(object):
    def __init__(self):

        rospy.init_node('waypoint_updater')
        rospy.logdebug('WaypointUpdater: __init__ starting')

        # Initialize local variables
        self._base_waypoints = None
        self._current_pose = None
        self.red_light_wp = -1
        self.current_linear_velocity = -1.
        self.speed_limit = rospy.get_param('/waypoint_loader/velocity') / 3.6  # m/s
        self.car_state = "go"  # Possible states will be: go, stop, idle

        # Choices are "constant" and "gradual"
        # "constant" sets all waypoints ahead to the desired speed
        # "gradual" linearly increases the waypoint speed to the desired speed
        self.go_mode = "gradual"
        self.stop_point = None  # The waypoint by which the car needs to stop

        # Choices are "slam" and "gradual"
        # "slam" sets all the waypoints to a speed of zero once car goes into stop state
        # "gradual" linearly decreases the waypoint speed until it reaches zero at the stop_point
        self.stop_mode = "gradual"
        self.stop_test = "no"  # Whether to test stopping at a particular waypoint rather than at stop lights
        self.max_acc = 1.  # in m/s^2

        # choices are self.find_nearest_basic and self.find_nearest
        # self.find_nearest_basic is borrowed from Udacity lessons
        # self.nearest is Paul's more sophisticated method
        self.find_nearest_wp = self.find_nearest_basic

        # Current position:
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        # Current velocity
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        # Base waypoints: are for the entire track and are only published once
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Final waypoints: First waypoint listed is the one directly in front of the car.
        #                  Total number of way points to include are given above by LOOKAHEAD_WPS
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # current waypoint index of car
        self.waypoint_index_pub = rospy.Publisher('/current_waypoint', Int32, queue_size=1)

        # TODO: Add other member variables you need below

        #rospy.spin()
        self.loop()

    # Looping instead of spinning
    def loop(self):
        rate = rospy.Rate(10)
        # rospy.logdebug("starting loop with car_state=%s and publishing rate=%s ", self.car_state, rate)

        # Loop when certain conditions are met
        while not rospy.is_shutdown():
            if (self._base_waypoints is None) or (self._current_pose is None):
                continue

            ####################################################################
            #  PART 0: SETUP
            ####################################################################

            # Find the nearest waypoint
            self._nearest = self.find_nearest_wp(self._current_pose, self._base_waypoints)
            rospy.logdebug("nearest waypoint index = %s", self._nearest)

            # Create the lane object to be published as final_waypoints
            myLane = Lane()

            # Create its header
            myLane.header.seq = 0
            myLane.header.stamp = rospy.Time(0)
            myLane.header.frame_id = '/world'

            # Initialize variables
            wp = []  # List of the waypoint indices corresponding to i
            myLane.waypoints = []

            # Loop through the waypoints to create/update
            for i in range(LOOKAHEAD_WPS):

                # Populate the waypoint list
                if i == 0:
                    # The first waypoint is just the nearest
                    index = self._nearest

                    # publish the nearest waypoint index
                    self.waypoint_index_pub.publish(Int32(index))
                else:
                    # Then we increment from there
                    index = self.next_waypoint(index, len(self._base_waypoints))
                wp.append(index)

                # Copy in the relevant waypoint
                myLane.waypoints.append(self._base_waypoints[index])

                # But modify the sequence number to contain the index number... will use later in tl_detector.py
                myLane.waypoints[i].pose.header.seq = index

            ####################################################################
            #  PART 1: FINITE STATE MACHINE -- SET THE STATE
            ####################################################################

            # From the "go" state we can only stop
            if self.car_state == "go":

                # If in testing mode
                if self.stop_test == "yes":

                    # Stop at waypoint 400
                    if wp[0] == 400:
                        self.car_state = "stop"
                        self.stop_point = 450

                # If not in testing mode
                else:

                    # Stop for real if there is a red light
                    if self.red_light_wp > -1:
                        # Check the deceleration needed to stop for it
                        total_dist = self.distance(self._base_waypoints, wp[0], self.red_light_wp)

                        # calculate the distance at which to safely start applying brakes
                        # using equation of motion v^2 = u^2 + 2 * a * s
                        # v = final velocity = 0
                        # u = initial velocity = v_max
                        # a =  - deceleration
                        # s = distance (to be calculated)
                        v_max = max(self.current_linear_velocity, self.speed_limit)
                        safe_braking_dist = max(SAFE_DIST, .5 * v_max * v_max / EASY_DECEL)

                        # using same equation, calculate the actual deceleration required
                        # if we are at or have already passed the safe braking distance
                        deceleration = 0.
                        if 0. < total_dist < safe_braking_dist:
                            v = self.current_linear_velocity
                            deceleration = .5 * v * v / total_dist
                        rospy.logdebug("There is a red light ahead: total_dist=%s, deceleration=%s",
                                      total_dist, deceleration)

                        # If the deceleration required is beyond a threshold, we don't stop!
                        # This covers scenario when we are very close to stop line
                        # and the light changes from green to yellow, we just continue
                        if EASY_DECEL < deceleration < HARD_DECEL:
                            self.car_state = "stop"
                            self.stop_point = self.red_light_wp

            # From the "stop" state we can either go to "idle" or "go"
            if self.car_state == "stop":

                # Go to idle when velocity is essentially zero
                if self.current_linear_velocity <= 0.01:
                    self.car_state = "idle"

                # Go back to go if there is no red light ahead
                elif self.red_light_wp < 0:
                    self.car_state = "go"

            # From the "idle" state we can only return to "go" once the light turns green
            if (self.car_state == "idle") and (self.red_light_wp < 0):
                self.car_state = "go"

            if False:
                rospy.logdebug("car_state= %s, current_linear_velocity=%s",
                               self.car_state, self.current_linear_velocity)

            ####################################################################
            #  PART 2: SET VELOCITY IN "go" STATE
            ####################################################################

            # Set velocity in the "go" state
            if self.car_state == "go":

                if False:
                    rospy.logdebug("************ %s",
                                  self.get_waypoint_velocity(myLane.waypoints[-1]))

                # Constant mode : we set the speed to speed limit
                if self.go_mode == "constant":
                    for i in range(LOOKAHEAD_WPS):
                        self.set_waypoint_velocity(myLane.waypoints, i, self.speed_limit)
                # Gradual mode: increment the velocity by max acc until it exceeds the target speed
                # using equation of motion : v^2 = u^2 + 2 * a * s
                # v = final velocity (just below speed limit)
                # u = initial velocity
                # a = max acceleration = self.max_acc
                # s = distance travelled
                if self.go_mode == "gradual":
                    # default target velocity
                    target_velocity = self.get_waypoint_velocity(self._base_waypoints[wp[-1]])

                    # Increment the velocity at nearest waypoint to 10% of current velocity
                    # or at least 1 mph, but making sure to stay within the speed limit
                    # rospy.logdebug("current velocity is %s", self.current_linear_velocity)
                    v = self.current_linear_velocity
                    v = max(1.1 * v, v + ONE_MPH)
                    v = min(v, self.speed_limit - ONE_MPH)

                    # Loop
                    for i in range(LOOKAHEAD_WPS):

                        # Set the new velocity
                        self.set_waypoint_velocity(myLane.waypoints, i, v)

                        # Calculate distance between successive waypoints
                        dist = 0.
                        if i < LOOKAHEAD_WPS-1:
                            dist = self.distance(self._base_waypoints, wp[i], wp[i+1])

                        v_sq = v * v + 2. * self.max_acc * dist
                        v = math.sqrt(max(0., v_sq))
                        v = min(v, self.speed_limit - ONE_MPH)

                        # Logging
                        # rospy.logdebug("velocity of waypoint %s (i=%s) set to %s",
                        #                wp[i], i, new_velocity)

            ####################################################################
            #  PART 3: SET VELOCITY IN "stop" STATE
            ####################################################################

            # Set velocity in the "stop" state
            elif self.car_state == "stop":

                # Slam mode
                if self.stop_mode == "slam":
                    for i in range(LOOKAHEAD_WPS):
                        self.set_waypoint_velocity(myLane.waypoints, i, 0.0)

                # Gradual mode: increment the velocity by self.max_accelleration until it exceeds the target speed
                elif self.stop_mode == "gradual":

                    # Calculate distance to the stop point
                    total_dist = self.distance(self._base_waypoints, wp[0], self.stop_point)
                    v_max = min(self.current_linear_velocity, self.speed_limit)
                    safe_braking_dist = max(SAFE_DIST, .5 * v_max * v_max / EASY_DECEL)

                    deceleration = 0.
                    v = self.current_linear_velocity
                    # Calculate deceleration needed
                    # add an extra deceleration (= EASY_DECEL) for safety
                    # also reduce the speed by 90% or at least 1 mph for the nearest waypoint
                    if 0. < total_dist < safe_braking_dist:
                        deceleration = .5 * v * v / total_dist + EASY_DECEL
                        v = max(v - ONE_MPH, 0.9 * v)
                    # rospy.logdebug("decelleration=%s",decelleration)

                    # Initialize flag indicating when the velocity must be zero
                    target_reached = False

                    # Loop and define the velocity at each point
                    for i in range(LOOKAHEAD_WPS):
                        # Check if we have reached the stop point
                        if wp[i] == self.stop_point:
                            target_reached = True

                        # Calculate distance between successive waypoints
                        dist = 0.
                        if i > 0:
                            dist = self.distance(self._base_waypoints, wp[i-1], wp[i])

                        # Calculate the new velocity
                        if target_reached:
                            v = 0.
                        else:
                            v_sq = v*v - 2. * deceleration * dist

                            # Floor the velocity at zero
                            if v_sq <= 0:
                                v = 0.
                            else:
                                v = math.sqrt(v_sq)

                        # Set the new velocity
                        self.set_waypoint_velocity(myLane.waypoints, i, v)
                        # Logging
                        # rospy.logdebug("velocity of waypoint %s (i=%s) set to %s",
                        #                wp[i], i, new_velocity)

            ####################################################################
            #  PART 4: SET VELOCITY IN "idle" STATE
            ####################################################################

            # Set velocity in the "idle" state
            elif self.car_state == "idle":
                for i in range(LOOKAHEAD_WPS):
                    self.set_waypoint_velocity(myLane.waypoints, i, 0.0)

            # Finally, publish the new velocities
            self.final_waypoints_pub.publish(myLane)

            # Wait a little before publishing the next command
            rate.sleep()

    def pose_cb(self, msg):
        #rospy.logdebug('WaypointUpdater: pose_cb starting')

        # Get the current position
        self._current_pose = msg.pose
        #rospy.logdebug("_current_pose: %s", self._current_pose)

    # Gets the waypoints
    def waypoints_cb(self, waypoints):
        #rospy.logdebug('WaypointUpdater: waypoints_cb starting')
        self._base_waypoints = waypoints.waypoints
        self.n_base_wps = len(self._base_waypoints)

    # Callback function for current_velocity
    def current_velocity_cb(self, msg):
        self.current_linear_velocity = msg.twist.linear.x # meters per second

    # Find the nearest waypoint (basic algorithm borrowed from Udacity lesson)
    def find_nearest_basic(self, curr_pose, base_wps):
        # Get the current x, y positions
        curr_x = curr_pose.position.x
        curr_y = curr_pose.position.y
        curr_yaw = self.get_yaw(curr_pose.orientation)
        # rospy.logdebug("current position (x,y) = (%s,%s)" , curr_x, curr_y)

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

    # Find the nearest waypoints...
    # Rough idea is to check what waypoint the car is closest to.
    # Also get the next and previous waypoints relative to that nearest one.
    # Then imagine car drives forward by a very small amount.  Then check
    # whether car got closer to the nearest waypoint, the next one, or
    # the previous one.  If it didn't get closer to any, then just return
    # nearest waypoint index.  Otherwise, return the waypoint car got closer
    # to by the most.  We are sure not to drive past any waypoint by only
    # moving by a very small amount
    def find_nearest(self, cur, base):

        # Get the current x, y positions
        cur_x = cur.position.x
        cur_y = cur.position.y
        # rospy.logdebug("current position (x,y) = (%s,%s)" , cur_x, cur_y)

        # Get number of waypoints on the map
        nWp = len(base)
        minpt = -1

        # Find the nearest waypoint
        for i in range(nWp):
            base_x = base[i].pose.pose.position.x
            base_y = base[i].pose.pose.position.y
            dist = self.dist(cur_x,base_x,cur_y,base_y)
            if i == 0:
                mindist = dist
                minpt=0
            else:
                if dist < mindist:
                    mindist=dist
                    minpt=i

        # Get the next and previous waypoint indices
        nextpt = self.next_waypoint( minpt, nWp )
        prevpt = self.next_waypoint( minpt, nWp )

        # Get the distance from the closest waypoint to the next and previous waypoints
        d_wp_next = self.dist(base[minpt].pose.pose.position.x, base[nextpt].pose.pose.position.x,
                              base[minpt].pose.pose.position.x, base[nextpt].pose.pose.position.x)
        d_wp_prev = self.dist(base[minpt].pose.pose.position.x, base[prevpt].pose.pose.position.x,
                              base[minpt].pose.pose.position.x, base[prevpt].pose.pose.position.x)

        # Get the distance from the car to the three (nearest, next, and previous) waypoints
        d_car_near = self.dist(cur_x, base[minpt].pose.pose.position.x,
                               cur_y, base[minpt].pose.pose.position.y)
        d_car_next = self.dist(cur_x, base[nextpt].pose.pose.position.x,
                               cur_y, base[nextpt].pose.pose.position.y)
        d_car_prev = self.dist(cur_x, base[prevpt].pose.pose.position.x,
                               cur_y, base[prevpt].pose.pose.position.y)

        # Get the distance to move -- minimum of all the previous distances calculated
        d_move = min( d_wp_next, d_wp_prev, d_car_near, d_car_next, d_car_prev )

        # Get orientation in euler angles
        q_w = cur.orientation.w
        q_x = cur.orientation.x
        q_y = cur.orientation.y
        q_z = cur.orientation.z
        roll = self.quaternion_to_euler_angle(q_w, q_x, q_y, q_z)
        # rospy.logdebug("roll = %s" , roll )

        # Project the car forward
        new_x, new_y = self.project_fwd(cur_x, cur_y, roll, d_move)

        # Calculate new distances from car to waypoints
        d_new_near = self.dist(new_x, base[minpt].pose.pose.position.x,
                               new_y, base[minpt].pose.pose.position.y)
        d_new_next = self.dist(new_x, base[nextpt].pose.pose.position.x,
                               new_y, base[nextpt].pose.pose.position.y)
        d_new_prev = self.dist(new_x, base[prevpt].pose.pose.position.x,
                               new_y, base[prevpt].pose.pose.position.y)

        # Calculate differences in distances... which waypoint, if any, did we
        # get closer to by driving forward a small amount?
        d_dif_near = d_car_near - d_new_near
        d_dif_next = d_car_next - d_new_next
        d_dif_prev = d_car_prev - d_new_prev

        # Check which waypoint we got closer to, if any.
        if d_dif_near < 0 and d_dif_next < 0 and d_dif_prev < 0:
            # Degnerate case... the waypoints get further away by driving
            # so just return the closest waypoint
            return minpt
        # Normal case where we get closer to one of the waypoints
        else:
            # Initialize to the nearest point
            maxdif = d_dif_near
            maxind = minpt
            # Check
            if d_dif_next > maxdif:
                maxdif = d_dif_next
                maxind = nextpt
            if d_dif_prev > maxdif:
                maxdif = d_dif_prev
                maxind = prevpt
            return maxind

    # Convert quaternion to euler angle.  Using formulas from this wiki page:
    #   https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # Not sure if roll, pitch, and yaw as x, y, and z are correctly labeled
    # Empirically what roll or X means is as follows.  The (x,y) coordinate system
    # has the quadrants as follows (moving counter clockwise):
    #   1. Upper right: +180 to +90
    #   2. Upper left: +90 to 0
    #   3. Lower left: 0 to -90
    #   4. Lower right: -90 to -180
    def quaternion_to_euler_angle(self, x, y, z, w):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.degrees(math.atan2(t0, t1))

        #t2 = +2.0 * (w * y - z * x)
        #t2 = +1.0 if t2 > +1.0 else t2
        #t2 = -1.0 if t2 < -1.0 else t2
        #Y = math.degrees(math.asin(t2))

        #t3 = +2.0 * (w * z + x * y)
        #t4 = +1.0 - 2.0 * (ysqr + z * z)
        #Z = math.degrees(math.atan2(t3, t4))

        return X #, Y, Z

    # Project the car forward.  See description of quadrants above.
    def project_fwd( self, x, y, roll, dist ):
        epsilon = 0.1
        # First handle corner cases where the angle is almost exactly
        # +180, +90, 0, -90, or -180
        if (180-epsilon < roll < 180) or (-180 < roll < -180+epsilon):
            new_x = x+dist
            new_y = y
        elif ( 90-epsilon < roll <  90+epsilon):
            new_x = x
            new_y = y+dist
        elif (  0-epsilon < roll <   0+epsilon):
            new_x = x-dist
            new_y = y
        elif (-90-epsilon < roll < -90+epsilon):
            new_x = x
            new_y = y-dist
        # Next handle normal cases for the four quadrants
        elif ( -180 <= roll <= 180):
            # Get the reference angle
            if 90 <= roll < 180:
              ref = math.radians(180-roll)
            elif 0 <= roll <  90:
              ref = math.radians(90-roll)
            elif -90 <= roll < 0:
              ref = math.radians(0-roll)
            elif -180 <= roll < -90:
              ref = math.radians(-90-roll)

            # Calculate the new coordinates
            new_x = dist*math.cos(ref)
            new_y = dist*math.sin(ref)
        # Error condition -- just return the originals
        else:
            new_x = x
            new_y = y
        return new_x, new_y

    # Get the next (+1) waypoint
    def next_waypoint( self, cur, n ):
        # If this is the last waypoint, start over at zero
        if cur == n - 1: nextwp = 0
        # Otherwise, increment by one
        else: nextwp = cur + 1
        return nextwp

    # Get the prev (-1) waypoint
    def prev_waypoint( self, cur, n ):
        # If this is the first waypoint, go to the last one
        if cur == 0: prevwp = n-1
        # Otherwise, decrement by one
        else: prevwp = cur + -1
        return prevwp

    def traffic_cb(self, msg):
        self.red_light_wp = msg.data
        # rospy.logdebug("waypoint_updater:traffic_cb says there is a red light at waypoint %s" , self.red_light_wp )

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # Euclidean distance.
    def dist(self, x1, x2, y1, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def distance(self, waypoints, wp1, wp2):
        dist = 0.
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
