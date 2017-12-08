from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
DA_MAX = 8.  # max change in acceleration allowed per sec.
BUFFER_SPEED = .7


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
        #self.max_torque = self.vehicle_mass * self.wheel_radius * accel_limit
        #self.min_torque = self.vehicle_mass * self.wheel_radius * decel_limit
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.prev_vels = []

        #self.throttle_pid = PID(1., 0, 0, self.min_torque, self.max_torque)
        self.steer_filter = LowPassFilter(0., 1.)
        self.dbw_enabled = False

    def control(self, tgt_linear, tgt_angular, cur_linear,
                cur_angular, dbw_enabled):
        if not dbw_enabled:
            if self.dbw_enabled:
                self.dbw_enabled = False
                self.steer_filter.reset()
                self.prev_vels = []
            return 0., 0., 0.

        # Get the angle from the yaw_controller
        yawcontroller = YawController(self.wheel_base, self.steer_ratio, ONE_MPH,
                                      self.max_lat_accel, self.max_steer_angle)
        steer_raw = yawcontroller.get_steering(tgt_linear, tgt_angular, cur_linear)
        steer = self.steer_filter.filt(steer_raw)
        self.prev_vels.append(cur_linear)
        if len(self.prev_vels) > 7:
            del self.prev_vels[0]

        # If dbw was just activated, we wait for the next call
        if not self.dbw_enabled:
            self.dbw_enabled = True
            return 0., 0., 0.

        dt = 0.02  # in seconds (~ 50 Hz)

        v = cur_linear
        curr_acc = (self.prev_vels[-1] - self.prev_vels[0]) / dt / len(self.prev_vels)
        acc = curr_acc
        gamma = abs(tgt_linear - cur_linear) / BUFFER_SPEED
        da = 0.1 * (1 + gamma)  # (jerk ~ 7.5 m/s^3)
        if v < tgt_linear:
            acc = max(da, acc + da)
            if v > tgt_linear - 0.25 * BUFFER_SPEED:
                acc = min(acc, da)
        else:
            acc = acc - da
        acc = min(acc, self.accel_limit)
        acc = max(acc, self.decel_limit)
        # drag force due to air
        F_drag = 0.4 * v * v
        # rolling resistance
        c_rr = .01 + 0.005 * pow(v / 28., 2)
        F_rr = c_rr * self.vehicle_mass * 9.8
        torque = (acc * self.vehicle_mass + F_drag + F_rr) * self.wheel_radius
        max_torque = 300.

        if acc > 0:
            throttle = min(.8, torque / max_torque)
            brake = 0.
        else:
            brake = max(0., -torque)
            throttle = 0.0

        if (False):
            rospy.loginfo("throttle : %s %s %s %s", throttle, curr_acc, cur_linear, tgt_linear)


        # Return throttle, brake, steer
        return throttle, brake, steer
