#!/usr/bin/env python3
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf
import math
import json
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# Import your existing implementations
from lab8_9_starter import Map, ParticleFilter, angle_to_neg_pi_to_pi  # :contentReference[oaicite:2]{index=2}
from lab10_starter import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD  # :contentReference[oaicite:3]{index=3}


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller while
         continuing to run the particle filter.
    """

    def __init__(self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state from odom / laser
        self.current_position: Optional[Dict[str, float]] = None
        self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        # Command publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        # PID controllers for tracking waypoints (copied from your ObstacleFreeWaypointController)
        self.linear_pid = WaypointPID(0.3, 0.0, 0.1, 10, -0.22, 0.22)
        self.angular_pid = WaypointPID(0.5, 0.0, 0.2, 10, -2.84, 2.84)

        # Waypoint tracking state
        self.plan: Optional[List[Dict[str, float]]] = None
        self.current_wp_idx: int = 0

        self.rate = rospy.Rate(10)

        # Wait until we have initial odom + scan
        while (self.current_position is None or self.laserscan is None) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ----------------------------------------------------------------------
    # Basic callbacks
    # ----------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # Use odom delta to propagate PF motion model
        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            # convert world delta to robot frame of previous pose
            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            # propagate all particles
            self._pf.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ----------------------------------------------------------------------
    # Low-level motion primitives
    # ----------------------------------------------------------------------
    def move_forward(self, distance: float):
        """
        Move the robot straight by a commanded distance (meters)
        using a constant velocity profile.
        """
        twist = Twist()
        speed = 0.15  # m/s
        twist.linear.x = speed if distance >= 0 else -speed

        duration = abs(distance) / speed if speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)

    def rotate_in_place(self, angle: float):
        """
        Rotate robot by a relative angle (radians).
        """
        twist = Twist()
        angular_speed = 0.8  # rad/s
        twist.angular.z = angular_speed if angle >= 0.0 else -angular_speed

        duration = abs(angle) / angular_speed if angular_speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ----------------------------------------------------------------------
    # Measurement update
    # ----------------------------------------------------------------------
    def take_measurements(self):
        """
        Use 3 beams (-15°, 0°, +15° in the robot frame) from /scan
        to update the particle filter via its measurement model.
        """
        if self.laserscan is None:
            return

        angle_min = self.laserscan.angle_min
        angle_increment = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        num_ranges = len(ranges)

        mid_idx = num_ranges // 2
        offset = int(15.0 / (angle_increment * 180.0 / math.pi))  # 15 degrees offset

        indices = [max(0, min(num_ranges - 1, mid_idx + i)) for i in (-offset, 0, offset)]
        measurements = []

        for idx in indices:
            z = ranges[idx]
            if z == inf or np.isinf(z):
                if hasattr(self.laserscan, "range_max"):
                    z = self.laserscan.range_max
                else:
                    z = 10.0  # fallback
            angle = angle_min + idx * angle_increment  # angle in robot frame
            measurements.append((z, angle))

        for z, a in measurements:
            self._pf.measure(z, a)

    # Helper
    def _pf_converged(self, std_threshold: float = 0.15) -> bool:
        """Return True when the particle cloud is tight enough."""
        xs = [p.x for p in self._pf._particles]
        ys = [p.y for p in self._pf._particles]
        return np.std(xs) < std_threshold and np.std(ys) < std_threshold

    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (explore a bit)
    # ----------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 400):
        """
        Simple autonomous exploration policy:
          - If front is free, go forward.
          - If obstacle close in front, back up and rotate.
        After each motion, apply PF measurement updates and check convergence.
        """
        
        ######### Your code starts here #########
        FRONT_CLEARANCE = 0.55
        FORWARD_DIST   = 0.4
        RANDOM_TURN_PROB = 0.10
 
        rate = rospy.Rate(10)
 
        for step in range(max_steps):
            if rospy.is_shutdown():
                break
 
            if self._pf_converged():
                rospy.loginfo(f"PF converged after {step} steps.")
                break
 
            xs = [p.x for p in self._pf._particles]
            ys = [p.y for p in self._pf._particles]
            rospy.loginfo(f"[PF] step {step:3d} | std_x={np.std(xs):.3f}  std_y={np.std(ys):.3f}")
 
            front_dist = inf
            if self.laserscan is not None:
                n = len(self.laserscan.ranges)
                front_idx = int((0.0 - self.laserscan.angle_min) / self.laserscan.angle_increment)
                front_idx = max(0, min(n - 1, front_idx))
                fd = self.laserscan.ranges[front_idx]
                if not (math.isnan(fd) or math.isinf(fd)):
                    front_dist = fd
 
            if front_dist < FRONT_CLEARANCE:
                turn = np.random.choice([pi / 2.0, -pi / 2.0])
                rospy.loginfo(f"  Wall ahead ({front_dist:.2f} m). Rotating {math.degrees(turn):.0f}")
                self.rotate_in_place(turn)
            elif np.random.rand() < RANDOM_TURN_PROB:
                turn = np.random.choice([pi / 2.0, -pi / 2.0])
                rospy.loginfo(f"  Random turn {math.degrees(turn):.0f}°")
                self.rotate_in_place(turn)
            else:
                self.move_forward(FORWARD_DIST)
 
            self.take_measurements()
            rate.sleep()
 
        self._stop()
 
        est_x, est_y, est_theta = self._pf.get_estimate()
        rospy.loginfo(
            f"PF estimate: x={est_x:.3f}  y={est_y:.3f}  θ={math.degrees(est_theta):.1f}"
        )
        ######### Your code ends here #########

        

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        est_x, est_y, est_theta = self._pf.get_estimate()
        start_position = {"x": est_x, "y": est_y, "theta": est_theta}
 
        rospy.loginfo(
            f"Planning from ({est_x:.3f}, {est_y:.3f}) → "
            f"({self.goal_position['x']:.3f}, {self.goal_position['y']:.3f})"
        )
 
        MAX_ATTEMPTS = 5
        for attempt in range(1, MAX_ATTEMPTS + 1):
            plan, graph = self._planner.generate_plan(start_position, self.goal_position)
 
            if plan:
                rospy.loginfo(f"  RRT succeeded on attempt {attempt} ({len(plan)} waypoints).")
                self.plan = plan
                self._planner.visualize_plan(plan)
                self._planner.visualize_graph(graph)
                return
 
            rospy.logwarn(f"  RRT attempt {attempt} failed — retrying…")
 
        raise RuntimeError(
            "RRT could not find a path."
        )
        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
        """
        Follow the RRT waypoints using PID on (distance, heading) error.
        Keep updating PF along the way.
        """
        ######### Your code starts here #########
        if not self.plan:
            rospy.logerr("follow_plan called but self.plan is empty.")
            return
 
        rate = rospy.Rate(20)
        ctrl_msg = Twist()
        current_wp_idx = 0
        PF_UPDATE_INTERVAL = 10
        tick = 0
 
        while not rospy.is_shutdown():
 
            if self.current_position is None:
                rate.sleep()
                continue
 
            if current_wp_idx >= len(self.plan):
                rospy.loginfo("All waypoints reached — stopping.")
                self._stop()
                break
 
            goal = self.plan[current_wp_idx]
 
            dx = goal["x"] - self.current_position["x"]
            dy = goal["y"] - self.current_position["y"]
            distance_error = sqrt(dx ** 2 + dy ** 2)
 
            target_theta = atan2(dy, dx)
            angle_error = angle_to_neg_pi_to_pi(
                target_theta - self.current_position["theta"]
            )
 
            t = rospy.get_time()
 
            linear_vel = self.linear_pid.control(distance_error, t)
            angular_vel = self.angular_pid.control(angle_error, t)
 
            if abs(angle_error) > 0.5:
                linear_vel = 0.0
 
            ctrl_msg.linear.x = linear_vel
            ctrl_msg.angular.z = angular_vel
            self.cmd_pub.publish(ctrl_msg)
 
            if distance_error < GOAL_THRESHOLD:
                rospy.loginfo(
                    f"  Reached waypoint {current_wp_idx + 1}/{len(self.plan)}"
                    f" ({goal['x']:.3f}, {goal['y']:.3f})"
                )
                current_wp_idx += 1
 
            tick += 1
            if tick % PF_UPDATE_INTERVAL == 0:
                self.take_measurements()
 
            rate.sleep()
 
        self._stop()
        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Top-level
    # ----------------------------------------------------------------------
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

    # Initialize ROS node
    rospy.init_node("pf_rrt_combined", anonymous=True)

    # Build map + PF + RRT
    map_obj = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.35

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    planner = RrtPlanner(obstacles, map_aabb)

    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
