#!/usr/bin/env python3
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf, sin, cos
from random import uniform
import math
import copy
import json
import numpy as np

import rospy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

from lab8_9_starter import (
    Map,
    ParticleFilter,
    Particle,
    PIDController,
    angle_to_neg_pi_to_pi,
)
from lab10_starter import RrtPlanner, ObstacleFreeWaypointController, GOAL_THRESHOLD


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller.
    """

    def __init__(
        self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]
    ):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state
        self.current_position: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        while (
            self.current_position is None or self.laserscan is None
        ) and not rospy.is_shutdown():
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    def odom_callback(self, msg: Odometry):
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

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    def forward_action(self, distance: float):
        pid_dist = PIDController(
            kP=1.2, kI=0.0, kD=1.5, kS=0.5, u_min=-0.22, u_max=0.22
        )
        pid_angle = PIDController(kP=1.2, kI=0.2, kD=1.0, kS=0.5, u_min=-2.0, u_max=2.0)

        start_pos = copy.deepcopy(self.current_position)
        start_theta = start_pos["theta"]

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            dx = self.current_position["x"] - start_pos["x"]
            dy = self.current_position["y"] - start_pos["y"]

            forward_progress = dx * math.cos(start_theta) + dy * math.sin(start_theta)
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
            self.cmd_pub.publish(twist)

            rate.sleep()

        self.cmd_pub.publish(Twist())

        delta_x = self.current_position["x"] - start_pos["x"]
        delta_y = self.current_position["y"] - start_pos["y"]
        delta_theta = angle_to_neg_pi_to_pi(
            self.current_position["theta"] - start_theta
        )

        self._pf.move_by(delta_x, delta_y, delta_theta)
        self._pf.visualize_particles()

    def rotate_action(self, goal_theta: float):
        pid = PIDController(kP=1.2, kI=0.2, kD=1.0, kS=0.5, u_min=-2.0, u_max=2.0)

        start_theta = self.current_position["theta"]
        target_theta = angle_to_neg_pi_to_pi(start_theta + goal_theta)

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            error = angle_to_neg_pi_to_pi(target_theta - self.current_position["theta"])

            if abs(error) < 0.03:
                break

            ang_vel = pid.control(error, rospy.get_time())
            self.cmd_pub.publish(Twist(angular=Vector3(z=ang_vel)))
            rate.sleep()

        self.cmd_pub.publish(Twist())

        actual_delta_theta = angle_to_neg_pi_to_pi(
            self.current_position["theta"] - start_theta
        )
        self._pf.move_by(0, 0, actual_delta_theta)
        self._pf.visualize_particles()

    def take_measurements(self):
        if self.laserscan is None:
            return

        selected_angles = [-135, -90, -45, 0, 45, 90, 135]

        for angle_deg in selected_angles:
            angle_rad = math.radians(angle_deg)

            idx = int(
                (angle_rad - self.laserscan.angle_min) / self.laserscan.angle_increment
            )

            if idx < 0 or idx >= len(self.laserscan.ranges):
                continue

            z = self.laserscan.ranges[idx]

            if z == float("inf") or math.isnan(z):
                continue

            self._pf.measure(z, angle_rad)

        self._pf.resample()
        self._pf.visualize_particles()
        self._pf.visualize_estimate()

    def localize_with_pf(self):
        rate = rospy.Rate(10)
        CONFIDENCE_THRESHOLD = 0.15
        RANDOM_TURN_PROB = 0.1
        MAX_CONVERGENCES = 1
        convergence_count = 0

        while not rospy.is_shutdown():

            particles_x = [p.x for p in self._pf._particles]
            particles_y = [p.y for p in self._pf._particles]

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

                rospy.loginfo(
                    "Reinitializing particles for next convergence attempt..."
                )
                x_min, x_max = self._pf._map.map_aabb[0], self._pf._map.map_aabb[1]
                y_min, y_max = self._pf._map.map_aabb[2], self._pf._map.map_aabb[3]
                new_particles = []
                spawned = 0
                while spawned < self._pf.n_particles:
                    x = uniform(x_min, x_max)
                    y = uniform(y_min, y_max)
                    if self._pf._is_invalid_position(x, y):
                        continue
                    theta = uniform(-pi, pi)
                    new_particles.append(Particle(x, y, theta, 0.0))
                    spawned += 1
                self._pf._particles = new_particles
                self._pf.visualize_particles()

            front_idx = int(
                (0.0 - self.laserscan.angle_min) / self.laserscan.angle_increment
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

        self.cmd_pub.publish(Twist())

    def plan_with_rrt(self):
        est_x, est_y, _ = self._pf.get_estimate()
        start = {"x": est_x, "y": est_y}

        rospy.loginfo(f"[RRT] start (PF)  x={est_x:.3f}  y={est_y:.3f}")
        rospy.loginfo(
            f"[RRT] goal        x={self.goal_position['x']:.3f}  y={self.goal_position['y']:.3f}"
        )

        self.plan, graph = self._planner.generate_plan(start, self.goal_position)

        if not self.plan:
            rospy.logwarn("[RRT] PF-estimate start failed — retrying with raw odom.")
            odom_start = {
                "x": self.current_position["x"],
                "y": self.current_position["y"],
            }
            self.plan, graph = self._planner.generate_plan(
                odom_start, self.goal_position
            )

        if not self.plan:
            rospy.logerr("[RRT] No plan found. Aborting.")
            return

        rospy.loginfo(f"[RRT] Plan found — {len(self.plan)} waypoints.")
        self._planner.visualize_plan(self.plan)
        self._planner.visualize_graph(graph)

    def follow_plan(self):
        if not self.plan:
            rospy.logwarn("[Follow] No plan — skipping.")
            return

        rospy.loginfo(f"=== Phase 3: Following {len(self.plan)} waypoints ===")

        from lab10_starter import PIDController as Lab10PID

        linear_pid = Lab10PID(kP=1.2, kI=0.00, kD=1.5, kS=0.5, u_min=0.0, u_max=0.22)
        angular_pid = Lab10PID(kP=1.2, kI=0.20, kD=1.0, kS=0.5, u_min=-2.0, u_max=2.0)

        rate = rospy.Rate(20)
        ctrl_msg = Twist()
        current_waypoint_idx = 0

        while not rospy.is_shutdown():

            if self.current_position is None:
                rate.sleep()
                continue

            if current_waypoint_idx >= len(self.plan):
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.cmd_pub.publish(ctrl_msg)
                break

            goal = self.plan[current_waypoint_idx]

            dx = goal["x"] - self.current_position["x"]
            dy = goal["y"] - self.current_position["y"]
            distance_error = sqrt(dx**2 + dy**2)
            target_theta = atan2(dy, dx)
            angle_error = target_theta - self.current_position["theta"]
            angle_error = atan2(sin(angle_error), cos(angle_error))

            t = rospy.get_time()
            linear_vel = linear_pid.control(distance_error, t)
            angular_vel = angular_pid.control(angle_error, t)

            if abs(angle_error) > 0.5:
                linear_vel = 0.0

            ctrl_msg.linear.x = linear_vel
            ctrl_msg.angular.z = angular_vel
            self.cmd_pub.publish(ctrl_msg)

            if distance_error < GOAL_THRESHOLD:
                current_waypoint_idx += 1

            rate.sleep()

    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    rospy.init_node("pf_rrt_combined", anonymous=True)

    map_obj = Map(obstacles, map_aabb)
    pf = ParticleFilter(
        map_obj,
        n_particles=200,
        translation_variance=0.003,
        rotation_variance=0.03,
        measurement_variance=0.1,
    )
    planner = RrtPlanner(obstacles, map_aabb)
    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
