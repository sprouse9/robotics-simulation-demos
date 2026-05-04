import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import open3d as o3d


@dataclass
class Config:
    # Link lengths
    l1: float = 0.9
    l2: float = 0.9
    l3: float = 0.30  # end-effector/tool length

    # Drawing plane above the robot
    plane_z: float = -1.0
    plane_width: float = 1.5
    plane_height: float = 1.0

    # Path density / speed
    row_spacing: float = 0.01
    samples_per_long_segment: int = 120
    samples_per_short_segment: int = 14
    dt: float = 0.02

    # Visualization threshold
    high_speed_threshold: float = 6

    # Start paused so you can orbit first
    start_paused: bool = True


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def forward_kinematics(q0: float, q1: float, q2: float, q3: float, cfg: Config) -> List[np.ndarray]:
    """
    Kinematic chain:
    q0 = base yaw
    q1 = shoulder pitch
    q2 = elbow pitch
    q3 = wrist pitch

    Returns:
    [base, shoulder, elbow, wrist, tool]
    """
    base = np.array([0.0, 0.0, 0.0])
    r0 = rot_z(q0)

    shoulder = base

    r1 = r0 @ rot_y(q1)
    elbow = shoulder + r1 @ np.array([cfg.l1, 0.0, 0.0])

    r2 = r1 @ rot_y(q2)
    wrist = elbow + r2 @ np.array([cfg.l2, 0.0, 0.0])

    r3 = r2 @ rot_y(q3)
    tool = wrist + r3 @ np.array([cfg.l3, 0.0, 0.0])

    return [base, shoulder, elbow, wrist, tool]


def inverse_kinematics(target: np.ndarray, cfg: Config) -> Tuple[float, float, float, float]:
    """
    IK with the final link constrained to be perpendicular to the horizontal plane.

    Since the robot is below the plane, we want the final link to point upward into the plane.
    In the radial-vertical plane, that means the final link should be vertical.

    target = [x, y, z]
    """
    x, y, z = float(target[0]), float(target[1]), float(target[2])

    # Base yaw aims the arm toward the XY projection of the target
    q0 = math.atan2(y, x)
    r = math.hypot(x, y)

    # Since l3 is vertical upward into the plane, solve the wrist point below the target
    wrist_r = r
    wrist_z = z + cfg.l3

    # Distance from shoulder/base to wrist point in the radial-vertical plane
    d = math.hypot(wrist_r, wrist_z)

    max_reach = cfg.l1 + cfg.l2 - 1e-6
    min_reach = abs(cfg.l1 - cfg.l2) + 1e-6
    d_clamped = min(max(d, min_reach), max_reach)

    # Elbow angle from law of cosines
    cos_q2 = (d_clamped**2 - cfg.l1**2 - cfg.l2**2) / (2.0 * cfg.l1 * cfg.l2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2 = math.acos(cos_q2)

    # Shoulder angle
    phi = math.atan2(wrist_z, wrist_r)
    psi = math.atan2(cfg.l2 * math.sin(q2), cfg.l1 + cfg.l2 * math.cos(q2))
    q1 = phi - psi

    # Force total pitch to +90 degrees so l3 is vertical upward
    q3 = -math.pi / 2.0 - (q1 + q2)

    return q0, q1, q2, q3


def generate_serpentine_path(cfg: Config) -> List[np.ndarray]:
    x0 = -cfg.plane_width / 2.0
    x1 =  cfg.plane_width / 2.0
    y0 = -cfg.plane_height / 2.0
    y1 =  cfg.plane_height / 2.0
    z = cfg.plane_z

    ys = []
    y = y0
    while y <= y1 + 1e-9:
        ys.append(y)
        y += cfg.row_spacing

    points: List[np.ndarray] = []

    for row_idx, yy in enumerate(ys):
        if row_idx % 2 == 0:
            xs = np.linspace(x0, x1, cfg.samples_per_long_segment)
        else:
            xs = np.linspace(x1, x0, cfg.samples_per_long_segment)

        for xx in xs:
            points.append(np.array([xx, yy, z], dtype=float))

        if row_idx < len(ys) - 1:
            next_y = ys[row_idx + 1]
            x_fixed = x1 if row_idx % 2 == 0 else x0
            connector_ys = np.linspace(yy, next_y, cfg.samples_per_short_segment)
            for cy in connector_ys[1:]:
                points.append(np.array([x_fixed, cy, z], dtype=float))

    return points


def create_plane(cfg: Config) -> o3d.geometry.TriangleMesh:
    # Thin box representing the drawing surface above the robot
    plane = o3d.geometry.TriangleMesh.create_box(
        width=cfg.plane_width,
        height=cfg.plane_height,
        depth=0.01
    )
    plane.translate(np.array([
        -cfg.plane_width / 2.0,
        -cfg.plane_height / 2.0,
        cfg.plane_z
    ]))
    plane.paint_uniform_color([0.82, 0.82, 0.82])
    plane.compute_vertex_normals()
    return plane


def create_frame() -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0.0, 0.0, 0.0])


def make_arm_lines(points: List[np.ndarray]) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(points))
    ls.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
    ], dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array([
        [0.10, 0.50, 0.90],
        [0.10, 0.50, 0.90],
        [0.10, 0.50, 0.90],
        [0.95, 0.55, 0.15],
    ]))
    return ls


def main() -> None:
    cfg = Config()
    path = generate_serpentine_path(cfg)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Robot Singularity Demo", width=2000, height=1200)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.06])
    opt.line_width = 10.0

    plane = create_plane(cfg)
    frame = create_frame()

    q_prev = np.array(inverse_kinematics(path[0], cfg), dtype=float)
    joints = forward_kinematics(*q_prev, cfg)
    arm = make_arm_lines(joints)

    trace_points: List[np.ndarray] = []
    trace_colors: List[np.ndarray] = []

    trace_lines = o3d.geometry.LineSet()
    trace_lines.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    trace_lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
    trace_lines.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

    vis.add_geometry(plane)
    vis.add_geometry(frame)
    vis.add_geometry(arm)
    vis.add_geometry(trace_lines)

    running = {"value": not cfg.start_paused}
    reset_flag = {"value": False}

    def toggle_run(_vis):
        running["value"] = not running["value"]
        print(f"Running: {running['value']}")
        return False

    def reset_demo(_vis):
        reset_flag["value"] = True
        print("Reset requested")
        return False

    vis.register_key_callback(ord(" "), toggle_run)
    vis.register_key_callback(ord("R"), reset_demo)

    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, 0.0, 0.85])
    ctr.set_front([1.0, -1.0, -0.8])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.6)

    idx = 0
    last_time = time.time()

    print("Controls:")
    print("  Space = start / pause")
    print("  R     = reset")
    print("  Mouse = orbit / zoom / pan using Open3D defaults")

    while True:
        if reset_flag["value"]:
            idx = 0
            q_prev = np.array(inverse_kinematics(path[0], cfg), dtype=float)
            joints = forward_kinematics(*q_prev, cfg)
            arm.points = o3d.utility.Vector3dVector(np.array(joints))

            trace_points.clear()
            trace_colors.clear()
            trace_lines.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            trace_lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            trace_lines.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

            vis.update_geometry(arm)
            vis.update_geometry(trace_lines)

            reset_flag["value"] = False
            running["value"] = not cfg.start_paused

        current_time = time.time()
        if running["value"] and current_time - last_time >= cfg.dt:
            if idx < len(path):
                target = path[idx]
                q = np.array(inverse_kinematics(target, cfg), dtype=float)

                dq = (q - q_prev) / cfg.dt
                speed_metric = float(np.linalg.norm(dq))

                joints = forward_kinematics(*q, cfg)
                arm.points = o3d.utility.Vector3dVector(np.array(joints))
                vis.update_geometry(arm)

                trace_points.append(joints[-1].copy())

                if speed_metric > cfg.high_speed_threshold:
                    color = np.array([0.65, 0.20, 0.90])  # purple
                else:
                    color = np.array([0.20, 0.85, 0.30])  # green

                if len(trace_points) >= 2:
                    trace_colors.append(color)
                    pts = np.array(trace_points)
                    segs = np.array([[i, i + 1] for i in range(len(trace_points) - 1)], dtype=np.int32)
                    cols = np.array(trace_colors)

                    trace_lines.points = o3d.utility.Vector3dVector(pts)
                    trace_lines.lines = o3d.utility.Vector2iVector(segs)
                    trace_lines.colors = o3d.utility.Vector3dVector(cols)
                    vis.update_geometry(trace_lines)

                if idx % 10 == 0:
                    print(
                        f"idx={idx:04d} "
                        f"target=({target[0]: .2f},{target[1]: .2f},{target[2]: .2f}) "
                        f"joint_speed={speed_metric: .2f} "
                        f"{'HIGH' if speed_metric > cfg.high_speed_threshold else ''}"
                    )

                q_prev = q
                idx += 2
            else:
                running["value"] = False
                print("Finished path")

            last_time = current_time

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


if __name__ == "__main__":
    main()