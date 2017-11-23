#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

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
        rospy.loginfo('WaypointUpdater: __init__ starting')

        # Initialize variable
        self.red_light_wp = -1
        self.current_linear_velocity = -1
        self.car_state = "go" # Possible states will be: go, stop, idle

        # Current position:
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        # Current velocity
        rospy.Subscriber('/current_velocity', TwistStamped , self.current_velocity_cb)

        # Base waypoints: are for the entire track and are only published once
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Final waypoints: First waypoint listed is the one directly in front of the car.
        #                  Total number of way points to include are given above by LOOKAHEAD_WPS
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        rospy.spin()

    def pose_cb(self, msg):
        #rospy.loginfo('WaypointUpdater: pose_cb starting')

        # Get the current position
        self._current_pose = msg.pose
        #rospy.loginfo("_current_pose: %s", self._current_pose)

        # Find the nearest waypoint
        self._nearest = self.find_nearest(self._current_pose , self._base_waypoints)
        rospy.loginfo("nearest waypoint index = %s", self._nearest)

        # Create the lane object to be published as final_waypoints
        myLane = Lane()

        # Create its header
        myLane.header.seq = 0
        myLane.header.stamp = rospy.Time(0)
        myLane.header.frame_id = 'WhatIsThisFor'

        # Create the waypoints locations
        myLane.waypoints = []
        last_wps = -1 # Last waypoint for use below
        for i in range(LOOKAHEAD_WPS):

          # The first waypoint is just the nearest
          if i==0:
            index = self._nearest
          # Then we increment from there -- no splines yet
          else:
            index = self.next_waypoint( index, len(self._base_waypoints) )
            if i==LOOKAHEAD_WPS-1: last_wps = index

          # Copy in the relevant waypoint
          myLane.waypoints.append(self._base_waypoints[index])

          # But modify the sequence number to contain the index number... will use later in tl_detector.py
          myLane.waypoints[i].pose.header.seq = index

        # Set velocities based on state...
        rospy.loginfo("car_state= %s",self.car_state)

        ## Set velocity in the "go" state -- other states will be added later
        if self.car_state == "go":

          # Accellerate to target velocity -- right now this is just a constant, but once the DBW module
          # is ready, this will accellerate gradually
          new_velocity = 30
          for i in range(LOOKAHEAD_WPS):
            self.set_waypoint_velocity(myLane.waypoints, i, new_velocity)

        # Finally, publish it
        self.final_waypoints_pub.publish(myLane)

    # Gets the waypoints
    def waypoints_cb(self, waypoints):
        #rospy.loginfo('WaypointUpdater: waypoints_cb starting')
        self._base_waypoints = waypoints.waypoints

    # Callback function for current_velocity
    def current_velocity_cb(self,msg):
        self.current_linear_velocity = msg.twist.linear.x # meters per second

    # Find the nearest waypoints...
    # Rough idea is to check what waypoint the car is closest to.
    # Also get the next and previous waypoints relative to that nearest one.
    # Then imagine car drives forward by a very small amount.  Then check
    # whether car got closer to the nearest waypoint, the next one, or
    # the previous one.  If it didn't get closer to any, then just return
    # nearest waypoint index.  Otherwise, return the waypoint car got closer
    # to by the most.  We are sure not to drive past any waypoint by only
    # moving by a very small amount
    def find_nearest(self,cur,base):

      # Get the current x, y positions
      cur_x = cur.position.x
      cur_y = cur.position.y
      rospy.loginfo("current position (x,y) = (%s,%s)" , cur_x, cur_y)

      # Get number of waypoints on the map
      nWp = len(base)

      # Find the nearest waypoint
      for i in range(nWp):
        base_x = base[i].pose.pose.position.x
        base_y = base[i].pose.pose.position.y
        dist = self.dist(cur_x,base_x,cur_y,base_y)
        if i==0:
          mindist=dist
          minpt=0
        else:
          if dist<mindist:
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
      d_car_near = self.dist( cur_x, base[minpt].pose.pose.position.x,
                              cur_y, base[minpt].pose.pose.position.y  )
      d_car_next = self.dist( cur_x, base[nextpt].pose.pose.position.x,
                              cur_y, base[nextpt].pose.pose.position.y )
      d_car_prev = self.dist( cur_x, base[prevpt].pose.pose.position.x,
                              cur_y, base[prevpt].pose.pose.position.y )

      # Get the distance to move -- minimum of all the previous distances calculated
      d_move = min( d_wp_next, d_wp_prev, d_car_near, d_car_next, d_car_prev )

      # Get orientation in euler angles
      q_w = cur.orientation.w
      q_x = cur.orientation.x
      q_y = cur.orientation.y
      q_z = cur.orientation.z
      roll = self.quaternion_to_euler_angle(q_w,q_x,q_y,q_z)
      #rospy.loginfo("roll = %s" , roll )

      # Project the car forward
      new_x, new_y = self.project_fwd( cur_x, cur_y, roll, d_move )

      # Calculate new distances from car to waypoints
      d_new_near = self.dist( new_x, base[minpt].pose.pose.position.x,
                              new_y, base[minpt].pose.pose.position.y  )
      d_new_next = self.dist( new_x, base[nextpt].pose.pose.position.x,
                              new_y, base[nextpt].pose.pose.position.y )
      d_new_prev = self.dist( new_x, base[prevpt].pose.pose.position.x,
                              new_y, base[prevpt].pose.pose.position.y )

      # Calculate differences in distances... which waypoint, if any, did we
      # get closer to by driving forward a small amount?
      d_dif_near = d_car_near - d_new_near
      d_dif_next = d_car_next - d_new_next
      d_dif_prev = d_car_prev - d_new_prev

      # Check which waypoint we got closer to, if any.
      if d_dif_near<0 and d_dif_next<0 and d_dif_prev<0:
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

    # Euclidean distance.  TODO: Find a common place to put this function
    def dist( self, x1, x2, y1, y2 ):
      dist = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
      return dist

    def traffic_cb(self, msg):
        self.red_light_wp = msg.data
        rospy.loginfo("waypoint_updater:traffic_cb says there is a red light at waypoint %s" , self.red_light_wp )

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
