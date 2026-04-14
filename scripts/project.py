#!/usr/bin/env python3
from typing import Dict
from argparse import ArgumentParser
import json

# Reuse lab8_9 classes exactly as-is — no reimplementation
from lab8_9_starter import (
    Map,
    ParticleFilter,
    Controller as Lab89Controller,
)

# Reuse lab10 classes exactly as-is — no reimplementation
from lab10_starter import (
    RrtPlanner,
    ObstacleFreeWaypointController,
)


def run(lab89_ctrl: Lab89Controller, planner: RrtPlanner, goal_position: Dict):
    """
    Three-phase pipeline:
      Phase 1 — Localise:   lab8_9 Controller.autonomous_exploration() verbatim.
      Phase 2 — Plan:       RrtPlanner.generate_plan() from lab10.
      Phase 3 — Navigate:   lab10 ObstacleFreeWaypointController.control_robot().
    """
    import rospy

    # ------------------------------------------------------------------
    # Phase 1: Particle-filter localisation
    # Calls the *exact* lab8_9 autonomous_exploration() — nothing rewritten.
    # ------------------------------------------------------------------
    lab89_ctrl.autonomous_exploration()

    # ------------------------------------------------------------------
    # Phase 2: RRT path planning from PF estimate → goal
    # ------------------------------------------------------------------
    est_x, est_y, _ = lab89_ctrl._particle_filter.get_estimate()
    start = {"x": est_x, "y": est_y}

    rospy.loginfo(f"[RRT] start (PF)  x={est_x:.3f}  y={est_y:.3f}")
    rospy.loginfo(
        f"[RRT] goal        x={goal_position['x']:.3f}  y={goal_position['y']:.3f}"
    )

    plan, graph = planner.generate_plan(start, goal_position)

    if not plan:
        rospy.logwarn("[RRT] PF-estimate start failed — retrying with raw odom.")
        odom_start = {
            "x": lab89_ctrl.current_position["x"],
            "y": lab89_ctrl.current_position["y"],
        }
        plan, graph = planner.generate_plan(odom_start, goal_position)

    if not plan:
        rospy.logerr("[RRT] No plan found. Aborting.")
        return

    rospy.loginfo(f"[RRT] Plan found — {len(plan)} waypoints.")
    planner.visualize_plan(plan)
    planner.visualize_graph(graph)

    # ------------------------------------------------------------------
    # Phase 3: Waypoint following
    # Uses the *exact* lab10 ObstacleFreeWaypointController — nothing rewritten.
    # ------------------------------------------------------------------
    waypoint_ctrl = ObstacleFreeWaypointController(plan)
    waypoint_ctrl.control_robot()


# =============================================================================
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

    map_obj = Map(obstacles, map_aabb)

    pf = ParticleFilter(
        map_obj,
        n_particles=200,
        translation_variance=0.1,   # matches lab8_9 main block
        rotation_variance=0.05,     # matches lab8_9 main block
        measurement_variance=0.1,   # matches lab8_9 main block
    )

    # NOTE: Lab89Controller.__init__ calls rospy.init_node internally.
    # Do NOT call rospy.init_node separately — it would cause a double-init error.
    lab89_ctrl = Lab89Controller(pf)

    planner = RrtPlanner(obstacles, map_aabb)

    try:
        run(lab89_ctrl, planner, goal_position)
    except Exception as e:
        import rospy
        rospy.logerr(f"Shutting down: {e}")