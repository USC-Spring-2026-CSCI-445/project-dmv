#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi
from time import sleep, time
import queue
import json
import math
from random import uniform
import copy

import scipy
import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import (
    Twist,
    Point32,
    PoseStamped,
    Pose,
    Vector3,
    Quaternion,
    Point,
    PoseArray,
)
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import scipy.stats
from numpy.random import choice

np.set_printoptions(linewidth=200)

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1

IMPOSSIBLE_LOG_P = -1e12
DEAD_THRESHOLD = -1e9


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


def angle_to_neg_pi_to_pi(angle: float) -> float:
    while angle < -pi:
        angle += 2 * pi
    while angle > pi:
        angle -= 2 * pi
    return angle


# see https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
def ray_line_intersection(ray_origin, ray_direction_rad, point1, point2):
    # Convert to numpy arrays
    ray_origin = np.array(ray_origin, dtype=np.float32)
    ray_direction = np.array(
        [math.cos(ray_direction_rad), math.sin(ray_direction_rad)]
    )
    point1 = np.array(point1, dtype=np.float32)
    point2 = np.array(point2, dtype=np.float32)

    # Ray-Line Segment Intersection Test in 2D
    v1 = ray_origin - point1
    v2 = point2 - point1
    v3 = np.array([-ray_direction[1], ray_direction[0]])
    denominator = np.dot(v2, v3)
    if denominator == 0:
        return None
    t1 = np.cross(v2, v1) / denominator
    t2 = np.dot(v1, v3) / denominator
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return [ray_origin + t1 * ray_direction]
    return None


class Map:
    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb

    @property
    def top_right(self) -> Tuple[float, float]:
        return self.map_aabb[1], self.map_aabb[3]

    @property
    def bottom_left(self) -> Tuple[float, float]:
        return self.map_aabb[0], self.map_aabb[2]

    def draw_distances(self, origins: List[Tuple[float, float]]):
        """Example usage:
        map_ = Map(obstacles, map_aabb)
        map_.draw_distances([(0.0, 0.0), (3, 3), (1.5, 1.5)])
        """

        # Draw scene
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()
        x_min_global, x_max_global, y_min_global, y_max_global = self.map_aabb
        for aabb in self.obstacles:
            width = aabb[1] - aabb[0]
            height = aabb[3] - aabb[2]
            rect = patches.Rectangle(
                (aabb[0], aabb[2]),
                width,
                height,
                linewidth=2,
                edgecolor="r",
                facecolor="r",
                alpha=0.4,
            )
            ax.add_patch(rect)
        ax.set_xlim(x_min_global, x_max_global)
        ax.set_ylim(y_min_global, y_max_global)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Plot of Obstacles")
        ax.set_aspect("equal", "box")
        plt.grid(True)

        # Draw rays
        angles = np.linspace(0, 2 * math.pi, 10, endpoint=False)
        for origin in origins:
            for angle in angles:
                closest_distance = self.closest_distance(origin, angle)
                if closest_distance is not None:
                    x = origin[0] + closest_distance * math.cos(angle)
                    y = origin[1] + closest_distance * math.sin(angle)
                    plt.plot([origin[0], x], [origin[1], y], "b-")
        plt.show()

    def closest_distance(
        self, origin: Tuple[float, float], angle: float
    ) -> Optional[float]:
        """Returns the closest distance to an obstacle from the given origin in the given direction `angle`. If no
        intersection is found, returns `None`.
        """

        def lines_from_obstacle(obstacle: OBS_TYPE):
            """Returns the four lines of the given AABB format obstacle.
            Example usage: `point0, point1 = lines_from_obstacle(self.obstacles[0])`
            """
            x_min, x_max, y_min, y_max = obstacle
            return [
                [(x_min, y_min), (x_max, y_min)],
                [(x_max, y_min), (x_max, y_max)],
                [(x_max, y_max), (x_min, y_max)],
                [(x_min, y_max), (x_min, y_min)],
            ]

        # Iterate over the obstacles in the map to find the closest distance (if there is one). Remember that the
        # obstacles are represented as a list of AABBs (Axis-Aligned Bounding Boxes) with the format
        # (x_min, x_max, y_min, y_max).
        result = None
        origin = np.array(origin)

        for obstacle in self.obstacles:
            for line in lines_from_obstacle(obstacle):
                p = ray_line_intersection(origin, angle, line[0], line[1])
                if p is None:
                    continue

                dist = np.linalg.norm(np.array(p) - origin)
                if result is None:
                    result = dist
                else:
                    result = min(result, dist)
        return result


# PID controller class
######### Your code starts here #########


class PIDController:
    """
    Generates control action taking into account instantaneous error (proportional action),
    accumulated error (integral action) and rate of change of error (derivative action).
    """

    def __init__(self, kP, kI, kD, kS, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        # initialize PID variables here
        ######### Your code starts here #########
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.u_min = u_min
        self.u_max = u_max
        self.err_prev = 0.0
        self.err_int = 0.0
        self.t_prev = None
        ######### Your code ends here #########

    def control(self, err, t):
        # compute PID control action here
        ######### Your code starts here #########
        if self.t_prev is None:
            self.t_prev = t
            return 0.0

        dt = t - self.t_prev
        if dt <= 1e-6:
            return 0.0

        derr = (err - self.err_prev) / dt

        self.err_int += err * dt
        self.err_int = max(-self.kS, min(self.kS, self.err_int))

        u = self.kP * err + self.kI * self.err_int + self.kD * derr
        u = max(self.u_min, min(self.u_max, u))

        self.err_prev = err
        self.t_prev = t
        return u


######### Your code ends here #########


class Particle:
    def __init__(self, x: float, y: float, theta: float, log_p: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.log_p = log_p

    def __str__(self) -> str:
        return (
            f"Particle<pose: {self.x, self.y, self.theta}, log_p: {self.log_p}>"
        )


class ParticleFilter:

    def __init__(
        self,
        map_: Map,
        n_particles: int,
        translation_variance: float,
        rotation_variance: float,
        measurement_variance: float,
    ):
        self.particles_visualization_pub = rospy.Publisher(
            "/pf_particles", PoseArray, queue_size=10
        )
        self.estimate_visualization_pub = rospy.Publisher(
            "/pf_estimate", PoseStamped, queue_size=10
        )

        # Initialize uniformly-distributed particles
        ######### Your code starts here #########
        self.map_ = map_
        self.n_particles = n_particles
        # save standard deviations for generating Gaussian Noise
        self.trans_std = math.sqrt(translation_variance)
        self.rot_std = math.sqrt(rotation_variance)
        self.meas_std = math.sqrt(measurement_variance)

        self._particles = []
        x_min, x_max, y_min, y_max = self.map_.map_aabb
        initial_log_p = math.log(
            1.0 / n_particles
        )  # each particle has equal initial probability

        for _ in range(n_particles):
            x = uniform(x_min, x_max)
            y = uniform(y_min, y_max)
            theta = uniform(-math.pi, math.pi)
            self._particles.append(Particle(x, y, theta, initial_log_p))
        ######### Your code ends here #########

    def visualize_particles(self):
        pa = PoseArray()
        pa.header.frame_id = "odom"
        pa.header.stamp = rospy.Time.now()
        for particle in self._particles:
            pose = Pose()
            pose.position = Point(particle.x, particle.y, 0.01)
            q_np = quaternion_from_euler(0, 0, float(particle.theta))
            pose.orientation = Quaternion(*q_np.tolist())
            pa.poses.append(pose)
        self.particles_visualization_pub.publish(pa)

    def visualize_estimate(self):
        ps = PoseStamped()
        ps.header.frame_id = "odom"
        ps.header.stamp = rospy.Time.now()
        x, y, theta = self.get_estimate()
        pose = Pose()
        pose.position = Point(x, y, 0.01)
        q_np = quaternion_from_euler(0, 0, float(theta))
        pose.orientation = Quaternion(*q_np.tolist())
        ps.pose = pose
        self.estimate_visualization_pub.publish(ps)

    def move_by(self, delta_x, delta_y, delta_theta):
        delta_theta = angle_to_neg_pi_to_pi(delta_theta)

        # Propagate motion of each particle
        ######### Your code starts here #########
        d = math.hypot(delta_x, delta_y)  # calculate actual movement distance
        x_min, x_max, y_min, y_max = self.map_.map_aabb # get map bounds

        for p in self._particles:
            # add Gaussian noise
            noisy_d = d + np.random.normal(0, self.trans_std) if d != 0 else 0
            noisy_delta_theta = (
                delta_theta + np.random.normal(0, self.rot_std)
                if delta_theta != 0
                else 0
            )

            # update particle state (Propagate)
            p.theta = angle_to_neg_pi_to_pi(p.theta + noisy_delta_theta)
            p.x += noisy_d * math.cos(p.theta)
            p.y += noisy_d * math.sin(p.theta)
            # ensure particles stay within map bounds
            p.x = max(x_min, min(p.x, x_max))
            p.y = max(y_min, min(p.y, y_max))
        ######### Your code ends here #########

    def measure(self, z: float, scan_angle_in_rad: float):
        """Update the particles based on the measurement `z` at the given `scan_angle_in_rad`.

        Args:
            z: distance to an obstacle
            scan_angle_in_rad: Angle in the robots frame where the scan was taken
        """

        # Calculate posterior probabilities and resample
        ######### Your code starts here #########
        log_weights = []

        for p in self._particles:
            # get the expected obstacle distance at that angle on the map
            global_angle = angle_to_0_to_2pi(p.theta + scan_angle_in_rad)
            expected_z = self.map_.closest_distance((p.x, p.y), global_angle)

            if expected_z is None:
                # if no obstacle is expected, assign a very low probability
                p.log_p += -100.0
            else:
                # calculate P(sensor reading | robot @ location) based on sensor noise Gaussian distribution
                log_prob = scipy.stats.norm.logpdf(
                    z, loc=expected_z, scale=self.meas_std
                )
                p.log_p += log_prob

            log_weights.append(p.log_p)

        log_weights = np.array(log_weights)

        # Log-sum-exp trick for numerical stability when converting log probabilities to normal probabilities
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        weights /= np.sum(weights)

        # Calculate effective sample size
        n_eff = 1.0 / np.sum(weights ** 2)
        # Calculate resampling threshold as half of the number of particles
        n_threshold = self.n_particles / 2.0

        if n_eff < n_threshold:
            # Resampling: based on weights (Roulette wheel) choose n_particles new particles
            # Low Variance Resampling incrementing by 1/(numParticles)
            new_particles = []
            new_log_p = math.log(1.0 / self.n_particles)

            # choose an initial random number r
            r = uniform(0, 1.0 / self.n_particles)
            c = weights[0]
            i = 0

            for m in range(self.n_particles):
                u = r + m * (1.0 / self.n_particles) # Incrementing by 1/N
                while u > c:
                    i += 1
                    c += weights[i]

                # copy the selected particle
                old_p = self._particles[i]
                new_particles.append(Particle(old_p.x, old_p.y, old_p.theta, new_log_p))

            self._particles = new_particles
        else:
          # If not resampling, just update the log probabilities
          for i, p in enumerate(self._particles):
              safe_weight = max(weights[i], 1e-300)
              p.log_p = math.log(safe_weight)
        ######### Your code ends here #########


class Controller:
    def __init__(self, particle_filter: ParticleFilter):
        rospy.init_node("particle_filter_controller", anonymous=True)
        self._particle_filter = particle_filter
        self._particle_filter.visualize_particles()

        #
        self.current_position = None
        self.laserscan = None
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.laserscan_sub = rospy.Subscriber(
            "/scan", LaserScan, self.robot_laserscan_callback
        )
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pointcloud_pub = rospy.Publisher(
            "/scan_pointcloud", PointCloud, queue_size=10
        )
        self.target_position_pub = rospy.Publisher(
            "/waypoints", MarkerArray, queue_size=10
        )

        while (
            (self.current_position is None) or (self.laserscan is None)
        ) and (not rospy.is_shutdown()):
            rospy.loginfo("waiting for odom and laserscan")
            rospy.sleep(0.1)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        self.current_position = {
            "x": pose.position.x,
            "y": pose.position.y,
            "theta": theta,
        }

    def robot_laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    def visualize_laserscan_ranges(self, idx_groups: List[Tuple[int, int]]):
        """Helper function to visualize ranges of sensor readings from the laserscan lidar.

        Example usage for visualizing the first 10 and last 10 degrees of the laserscan:
            `self.visualize_laserscan_ranges([(0, 10), (350, 360)])`
        """
        pcd = PointCloud()
        pcd.header.frame_id = "odom"
        pcd.header.stamp = rospy.Time.now()
        for idx_low, idx_high in idx_groups:
            for idx, d in enumerate(self.laserscan.ranges[idx_low:idx_high]):
                if d == inf:
                    continue
                angle = math.radians(idx) + self.current_position["theta"]
                x = d * math.cos(angle) + self.current_position["x"]
                y = d * math.sin(angle) + self.current_position["y"]
                z = 0.1
                pcd.points.append(Point32(x=x, y=y, z=z))
                pcd.channels.append(
                    ChannelFloat32(name="rgb", values=(0.0, 1.0, 0.0))
                )
        self.pointcloud_pub.publish(pcd)

    def visualize_position(self, x: float, y: float):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = Point(x, y, 0.0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Vector3(0.075, 0.075, 0.1)
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
        marker_array.markers.append(marker)
        self.target_position_pub.publish(marker_array)

    def take_measurements(self):
        if self.laserscan is None:
            return

        selected_angles = [0, 90]

        for angle_deg in selected_angles:
            angle_rad = math.radians(angle_deg)

            idx = int(
                (angle_rad - self.laserscan.angle_min)
                / self.laserscan.angle_increment
            )

            if idx < 0 or idx >= len(self.laserscan.ranges):
                continue

            z = self.laserscan.ranges[idx]

            if z == float("inf") or math.isnan(z):
                continue

            self._particle_filter.measure(z, angle_rad)

        self._particle_filter.resample()
        self._particle_filter.visualize_particles()
        self._particle_filter.visualize_estimate()

    def autonomous_exploration(self):
        """Randomly explore the environment here, while making sure to call `take_measurements()` and
        `_particle_filter.move_by()`. The particle filter should converge on the robots position eventually.

        Note that the following visualizations functions are available:
            visualize_position(...)
            visualize_laserscan_ranges(...)
        """
        # Robot autonomously explores environment while it localizes itself
        ######### Your code starts here #########
        rate = rospy.Rate(10)
        CONFIDENCE_THRESHOLD = 0.15
        RANDOM_TURN_PROB = 0.1
        MAX_CONVERGENCES = 1
        convergence_count = 0

        while not rospy.is_shutdown():

            particles_x = [p.x for p in self._particle_filter._particles]
            particles_y = [p.y for p in self._particle_filter._particles]

            std_x = np.std(particles_x)
            std_y = np.std(particles_y)

            rospy.loginfo(
                f"Spread - X: {std_x:.3f}, Y: {std_y:.3f} | Convergences: {convergence_count}/{MAX_CONVERGENCES}"
            )

            if std_x < CONFIDENCE_THRESHOLD and std_y < CONFIDENCE_THRESHOLD:
                convergence_count += 1
                rospy.loginfo(
                    f"Localization converged! ({convergence_count}/{MAX_CONVERGENCES})"
                )

                if convergence_count >= MAX_CONVERGENCES:
                    rospy.loginfo("Localization complete!")
                    break

                # Reinitialize particles uniformly so the filter tries again
                # from scratch. The robot has moved, so the new sensor context
                # gives a fresh chance to land on the true position.
                rospy.loginfo(
                    "Reinitializing particles for next convergence attempt..."
                )
                x_min, x_max = (
                    self._particle_filter._map.map_aabb[0],
                    self._particle_filter._map.map_aabb[1],
                )
                y_min, y_max = (
                    self._particle_filter._map.map_aabb[2],
                    self._particle_filter._map.map_aabb[3],
                )
                new_particles = []
                spawned = 0
                while spawned < self._particle_filter.n_particles:
                    x = uniform(x_min, x_max)
                    y = uniform(y_min, y_max)
                    if self._particle_filter._is_invalid_position(x, y):
                        continue
                    theta = uniform(-pi, pi)
                    new_particles.append(Particle(x, y, theta, 0.0))
                    spawned += 1
                self._particle_filter._particles = new_particles
                self._particle_filter.visualize_particles()

            front_idx = int(
                (0.0 - self.laserscan.angle_min)
                / self.laserscan.angle_increment
            )
            front_idx = max(0, min(len(self.laserscan.ranges) - 1, front_idx))
            front_dist = self.laserscan.ranges[front_idx]

            if math.isnan(front_dist) or (
                front_dist != float("inf") and front_dist < 0.55
            ):
                rospy.loginfo("Wall detected, re-routing...")
                turn = np.random.choice([pi / 2, -pi / 2])
                self.rotate_action(turn)

            else:
                if np.random.rand() < RANDOM_TURN_PROB:
                    rospy.loginfo("Random exploration turn")
                    turn = np.random.choice([pi / 2, -pi / 2])
                    self.rotate_action(turn)
                else:
                    self.forward_action(0.4)

            self.take_measurements()
            rate.sleep()

        self.robot_ctrl_pub.publish(Twist())
        ######### Your code ends here #########

    def forward_action(self, distance: float):
        # Robot moves forward by a set amount during manual control
        ######### Your code starts here #########
        pid_dist = PIDController(
            kP=1.2, kI=0.0, kD=1.5, kS=0.5, u_min=-0.22, u_max=0.22
        )
        pid_angle = PIDController(
            kP=1.2, kI=0.2, kD=1.0, kS=0.5, u_min=-2.0, u_max=2.0
        )

        start_pos = copy.deepcopy(self.current_position)
        start_theta = start_pos["theta"]

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            dx = self.current_position["x"] - start_pos["x"]
            dy = self.current_position["y"] - start_pos["y"]

            forward_progress = dx * math.cos(start_theta) + dy * math.sin(
                start_theta
            )
            distance_error = distance - forward_progress

            heading_error = angle_to_neg_pi_to_pi(
                start_theta - self.current_position["theta"]
            )

            if abs(distance_error) < 0.02:
                break

            v = pid_dist.control(distance_error, rospy.get_time())
            w = pid_angle.control(heading_error, rospy.get_time())

            twist = Twist()
            twist.linear.x = v
            twist.angular.z = w
            self.robot_ctrl_pub.publish(twist)

            rate.sleep()

        self.robot_ctrl_pub.publish(Twist())

        delta_x = self.current_position["x"] - start_pos["x"]
        delta_y = self.current_position["y"] - start_pos["y"]
        delta_theta = angle_to_neg_pi_to_pi(
            self.current_position["theta"] - start_theta
        )

        self._particle_filter.move_by(delta_x, delta_y, delta_theta)
        self._particle_filter.visualize_particles()
        ######### Your code ends here #########

    def rotate_action(self, goal_theta: float):
        pid = PIDController(
            kP=1.2, kI=0.2, kD=1.0, kS=0.5, u_min=-2.0, u_max=2.0
        )

        start_theta = self.current_position["theta"]
        target_theta = angle_to_neg_pi_to_pi(start_theta + goal_theta)

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            error = angle_to_neg_pi_to_pi(
                target_theta - self.current_position["theta"]
            )

            if abs(error) < 0.03:
                break

            ang_vel = pid.control(error, rospy.get_time())
            self.robot_ctrl_pub.publish(Twist(angular=Vector3(z=ang_vel)))
            rate.sleep()

        self.robot_ctrl_pub.publish(Twist())

        actual_delta_theta = angle_to_neg_pi_to_pi(
            self.current_position["theta"] - start_theta
        )
        self._particle_filter.move_by(0, 0, actual_delta_theta)
        self._particle_filter.visualize_particles()


""" Example usage

rosrun development lab8_9.py --map_filepath src/csci455l/scripts/lab8_9_map.json
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]

    map_ = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.1
    rotation_variance = 0.05
    measurement_variance = 0.1
    particle_filter = ParticleFilter(
        map_,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    controller = Controller(particle_filter)

    try:
        # # Manual control
        # goal_theta = 0
        # controller.take_measurements()
        # while not rospy.is_shutdown():
        #     print("\nEnter 'a', 'w', 's', 'd' to move the robot:")
        #     uinput = input("")
        #     if uinput == "w": # forward
        #         ######### Your code starts here #########
        #         controller.forward_action(0.5)
        #         ######### Your code ends here #########
        #     elif uinput == "a": # left
        #         ######### Your code starts here #########
        #         controller.rotate_action(pi/2)
        #         ######### Your code ends here #########
        #     elif uinput == "d": #right
        #         ######### Your code starts here #########
        #         controller.rotate_action(-pi/2)
        #         ######### Your code ends here #########
        #     elif uinput == "s": # backwards
        #         ######### Your code starts here #########
        #         controller.forward_action(-0.5)
        #         ######### Your code ends here #########
        #     else:
        #         print("Invalid input")
        #     ######### Your code starts here #########
        #     controller.take_measurements()
        #     ######### Your code ends here #########

        # Autonomous exploration
        ######### Your code starts here #########
        controller.autonomous_exploration()
        ######### Your code ends here #########

    except rospy.ROSInterruptException:
        print("Shutting down...")
