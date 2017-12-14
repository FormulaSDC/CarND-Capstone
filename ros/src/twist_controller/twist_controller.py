from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
DA_MAX = 8.  # max change in acceleration allowed per sec.


class Controller(object):

    def __init__(self, wheel_base, steer_ratio, max_lat_accel,
                 max_steer_angle, vehicle_mass, wheel_radius, fuel_capacity,
                 accel_limit, decel_limit):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        self.vehicle_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self.wheel_radius = wheel_radius
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.acc_pid = PID(2.5, 0., 0., -1., 1.)
        self.acc_filter = LowPassFilter(3., 1.)
        self.steer_filter = LowPassFilter(1., 1.)
        self.dbw_enabled = False

    def control(self, tgt_linear, tgt_angular, cur_linear, cur_angular, dbw_enabled):
        if not dbw_enabled:
            if self.dbw_enabled:
                self.dbw_enabled = False
                self.acc_pid.reset()
                self.acc_filter.reset()
                self.steer_filter.reset()
            return 0., 0., 0.

        dt = 0.02  # in seconds (~ 50 Hz)
        v = max(0., cur_linear)
        vel_error = tgt_linear - v

        # Get the angle from the yaw_controller
        yawcontroller = YawController(self.wheel_base, self.steer_ratio, ONE_MPH,
                                      self.max_lat_accel, self.max_steer_angle)
        steer_raw = yawcontroller.get_steering(tgt_linear, tgt_angular, cur_linear)
        steer = self.steer_filter.filt(steer_raw)
        acc_raw = self.acc_pid.step(vel_error, dt)
        acc = self.acc_filter.filt(acc_raw)

        # If dbw was just activated, we wait for the next call
        if not self.dbw_enabled:
            self.dbw_enabled = True
            return 0., 0., 0.

        # drag force due to air
        F_drag = 0.4 * v * v
        # rolling resistance
        c_rr = .01 + 0.005 * pow(v / 28., 2)
        F_rr = c_rr * self.vehicle_mass * 9.8
        torque = (acc * self.vehicle_mass + F_drag + F_rr) * self.wheel_radius
        max_torque = 1000.

        # there is a constant bias of throttle we need to correct for
        torque -= 0.02 * max_torque

        if acc > 0:
            throttle = min(.25, max(torque, 0.) / max_torque)
            brake = 0.
        else:
            brake = max(0., -torque)
            # for idle state, we apply a small constant brake :
            if tgt_linear < 0.01:
                brake = 10.
            throttle = 0.0

        # rospy.logdebug("throttle : %s %s %s %s", throttle, acc, cur_linear, brake)

        # Return throttle, brake, steer
        return throttle, brake, steer
