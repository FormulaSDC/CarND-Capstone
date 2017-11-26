from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MAX_ACC = 10.


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
        self.max_torque = self.vehicle_mass * self.wheel_radius \
                          * accel_limit
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit

        self.throttle_pid = PID(1., 1., 1.)
        self.steer_filter = LowPassFilter(0., 1.)
        self.dbw_enabled = False

    # This is just some starter code that only works at low speeds (<20 mph)
    def control(self, tgt_linear, tgt_angular, cur_linear,
                cur_angular, dbw_enabled):
        if not dbw_enabled:
            if self.dbw_enabled:
                self.dbw_enabled = False
                self.steer_filter.reset()
            return 0., 0., 0.

        # distance travelled to reach target speed
        ds = 20.
        acc = (pow(tgt_linear, 2) - pow(cur_linear, 2)) / (2 * ds)
        acc = min(acc, self.accel_limit)
        acc = max(acc, self.decel_limit)
        torque = acc * self.vehicle_mass * self.wheel_radius
        #rospy.loginfo("Torque : %s %s", torque, acc)
        if torque > 0:
            throttle = min(1., torque / self.max_torque)
            brake = 0.
        else:
            brake = -torque
            throttle = 0.0

        # Get the angle from the yaw_controller
        yawcontroller = YawController(self.wheel_base, self.steer_ratio, ONE_MPH,
                                      self.max_lat_accel, self.max_steer_angle)
        steer_raw = yawcontroller.get_steering(tgt_linear, tgt_angular, cur_linear)
        steer = self.steer_filter.filt(steer_raw)

        # If dbw was just activated, we wait for the next call
        if not self.dbw_enabled:
            self.dbw_enabled = True
            return 0., 0., 0.

        # Return throttle, brake, steer
        return throttle, brake, steer
