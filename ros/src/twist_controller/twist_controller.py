from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):

    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        self.throttle_pid = PID(1., 1., 1.)
        self.steer_filter = LowPassFilter(1., 2.)
        self.engaged = False


    def reset(self):
        self.steer_filter.reset()


    # This is just some starter code that only works at low speeds (<20 mph)
    def control( self, tgt_linear, tgt_angular, cur_linear, cur_angular ):

        # If going slower than the target, full throttle.  Otherwise, no throttle.
        if tgt_linear > cur_linear: throttle=1.0
        else: throttle=0.0

        # Get the angle from the yaw_controller
        self.yawcontroller = YawController(self.wheel_base, self.steer_ratio, self.min_speed,
                                           self.max_lat_accel, self.max_steer_angle)
        steer_raw = self.yawcontroller.get_steering(tgt_linear, tgt_angular, cur_linear)
        steer = steer_raw #self.steer_filter.filt(steer_raw)

        # Return throttle, brake, steer
        return throttle, 0.0, steer
