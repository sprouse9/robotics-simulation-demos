import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import open3d as o3d


@dataclass
class Config:
    # Robot link lengths
    l1: float = 0.9
    l2: float = 0.9
    l3: float = 0.30

    # Pipe settings
    pipe_radius: float = 0.28
    pipe_length: float = 1.4
    pipe_center_x: float = 1.15
    pipe_center_y: float = 0.0
    pipe_center_z: float = -0.55

    # Torch target location
    torch_clearance: float = 0.03

    # Animation
    dt: float = 0.02
    pipe_rad_per_sec: float = math.radians(45)

    # Optional tiny torch weave
    weave_enabled: bool = True
    weave_amplitude: float = 0.025
    weave_frequency_hz: float = 1.2

    high_speed_threshold: float = 1.0
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
    x, y, z = float(target[0]), float(target[1]), float(target[2])

    q0 = math.atan2(y, x)
    r = math.hypot(x, y)

    # Tool points downward toward top of pipe.
    # Therefore wrist is above the target by l3.
    wrist_r = r
    wrist_z = z + cfg.l3

    d = math.hypot(wrist_r, wrist_z)

    max_reach = cfg.l1 + cfg.l2 - 1e-6
    min_reach = abs(cfg.l1 - cfg.l2) + 1e-6
    d = min(max(d, min_reach), max_reach)

    cos_q2 = (d**2 - cfg.l1**2 - cfg.l2**2) / (2.0 * cfg.l1 * cfg.l2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))

    q2 = math.acos(cos_q2)

    phi = math.atan2(wrist_z, wrist_r)
    psi = math.atan2(cfg.l2 * math.sin(q2), cfg.l1 + cfg.l2 * math.cos(q2))
    q1 = phi - psi

    # Keep tool approximately vertical/down into pipe.
    q3 = math.pi / 2.0 - (q1 + q2)

    return q0, q1, q2, q3


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


def create_pipe(cfg: Config) -> o3d.geometry.TriangleMesh:
    pipe = o3d.geometry.TriangleMesh.create_cylinder(
        radius=cfg.pipe_radius,
        height=cfg.pipe_length,
        resolution=72,
        split=4,
    )

    # Open3D cylinder is along Z by default.
    # Rotate it so pipe axis runs along Y.
    pipe.rotate(pipe.get_rotation_matrix_from_xyz((math.pi / 2, 0.0, 0.0)), center=(0, 0, 0))

    pipe.translate([cfg.pipe_center_x, cfg.pipe_center_y, cfg.pipe_center_z])
    pipe.paint_uniform_color([0.45, 0.45, 0.48])
    pipe.compute_vertex_normals()
    return pipe


def create_weld_seam(cfg: Config) -> o3d.geometry.LineSet:
    # Circular seam around pipe at y = 0
    points = []
    lines = []

    n = 120
    for i in range(n):
        a = 2.0 * math.pi * i / n
        x = cfg.pipe_center_x + cfg.pipe_radius * math.cos(a)
        y = cfg.pipe_center_y
        z = cfg.pipe_center_z + cfg.pipe_radius * math.sin(a)
        points.append([x, y, z])
        lines.append([i, (i + 1) % n])

    seam = o3d.geometry.LineSet()
    seam.points = o3d.utility.Vector3dVector(np.array(points))
    seam.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    seam.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.35, 0.05], (n, 1)))
    return seam


def create_rotation_markers(cfg: Config, num_stripes: int = 18) -> o3d.geometry.LineSet:
    marker = o3d.geometry.LineSet()

    # placeholder arrays (will be updated every frame)
    marker.points = o3d.utility.Vector3dVector(np.zeros((num_stripes * 2, 3)))
    marker.lines = o3d.utility.Vector2iVector(
        np.array([[i * 2, i * 2 + 1] for i in range(num_stripes)], dtype=np.int32)
    )

    # color gradient around the pipe
    colors = []
    for i in range(num_stripes):
        t = i / num_stripes
        colors.append([
            0.5 + 0.5 * math.cos(2 * math.pi * t),
            0.5 + 0.5 * math.sin(2 * math.pi * t),
            0.8
        ])
    marker.colors = o3d.utility.Vector3dVector(np.array(colors))

    return marker


def update_markers(marker: o3d.geometry.LineSet, cfg: Config, base_angle: float) -> None:
    num_stripes = len(marker.lines)

    points = []

    for i in range(num_stripes):
        angle = base_angle + (2 * math.pi * i / num_stripes)

        x = cfg.pipe_center_x + cfg.pipe_radius * math.cos(angle)
        z = cfg.pipe_center_z + cfg.pipe_radius * math.sin(angle)

        y0 = cfg.pipe_center_y - cfg.pipe_length / 2.0
        y1 = cfg.pipe_center_y + cfg.pipe_length / 2.0

        p0 = [x, y0, z]
        p1 = [x, y1, z]

        points.extend([p0, p1])

    marker.points = o3d.utility.Vector3dVector(np.array(points))


def main() -> None:
    cfg = Config()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Pipe Rotator Weld Demo", width=2000, height=1200)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.06])
    opt.line_width = 8.0

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    pipe = create_pipe(cfg)
    seam = create_weld_seam(cfg)
    marker = create_rotation_markers(cfg, num_stripes=18)

    torch_target_base = np.array([
        cfg.pipe_center_x,
        cfg.pipe_center_y,
        cfg.pipe_center_z + cfg.pipe_radius + cfg.torch_clearance,
    ])

    q_prev = np.array(inverse_kinematics(torch_target_base, cfg), dtype=float)
    joints = forward_kinematics(*q_prev, cfg)
    arm = make_arm_lines(joints)

    vis.add_geometry(frame)
    vis.add_geometry(pipe)
    vis.add_geometry(seam)
    vis.add_geometry(marker)
    vis.add_geometry(arm)

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
    ctr.set_lookat([0.7, 0.0, -0.35])
    ctr.set_front([1.0, -1.2, -0.6])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.75)

    print("Controls:")
    print("  Space = start / pause")
    print("  R     = reset")
    print("  Mouse = orbit / zoom / pan")
    print()
    print("This demo shows positioner-assisted pipe welding:")
    print("  - pipe rotates")
    print("  - torch stays mostly fixed")
    print("  - robot joint speeds stay low")

    t = 0.0
    pipe_angle = 0.0
    last_time = time.time()
    step = 0

    while True:
        if reset_flag["value"]:
            t = 0.0
            pipe_angle = 0.0
            step = 0
            q_prev = np.array(inverse_kinematics(torch_target_base, cfg), dtype=float)
            update_markers(marker, cfg, pipe_angle)
            vis.update_geometry(marker)
            reset_flag["value"] = False
            running["value"] = not cfg.start_paused

        current_time = time.time()

        if running["value"] and current_time - last_time >= cfg.dt:
            t += cfg.dt
            pipe_angle += cfg.pipe_rad_per_sec * cfg.dt

            # Visually rotate only the marker.
            # The pipe mesh itself is symmetric, so rotating the full cylinder would not look different.
            update_markers(marker, cfg, pipe_angle)
            vis.update_geometry(marker)

            target = torch_target_base.copy()

            # Tiny weave to make it look like torch control, not a totally frozen robot.
            if cfg.weave_enabled:
                target[1] += cfg.weave_amplitude * math.sin(2.0 * math.pi * cfg.weave_frequency_hz * t)

            q = np.array(inverse_kinematics(target, cfg), dtype=float)
            dq = (q - q_prev) / cfg.dt
            speed_metric = float(np.linalg.norm(dq))

            joints = forward_kinematics(*q, cfg)
            arm.points = o3d.utility.Vector3dVector(np.array(joints))
            vis.update_geometry(arm)

            if step % 20 == 0:
                print(
                    f"step={step:04d} "
                    f"pipe_angle={math.degrees(pipe_angle) % 360:6.1f} deg "
                    f"joint_speed={speed_metric:6.3f} rad/s "
                    f"{'HIGH' if speed_metric > cfg.high_speed_threshold else ''}"
                )

            q_prev = q
            step += 1
            last_time = current_time

        if not vis.poll_events():
            break

        vis.update_renderer()

    vis.destroy_window()


if __name__ == "__main__":
    main()