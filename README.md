# Robotics Simulation Demos

Small Python robotics and automation simulation demos exploring robot motion, singularities, workpiece positioning, and welding-style automation concepts.

These demos were built as experimental engineering visualizations to understand robot kinematics, motion constraints, and how positioning systems can reduce robot joint motion during repetitive manufacturing operations.

## Demos

### 1. Robot Singularity Demo


https://github.com/user-attachments/assets/e444b5a6-7d9d-4653-9e75-4b9e938710d5



https://github.com/user-attachments/assets/1c89df8a-6986-4d93-95bd-119f41210e6f








**File:** `singularity_demo2.py`

Demonstrates robot arm motion and singularity-related behavior using a visual simulation. The goal is to explore how robot joint configurations, end-effector positioning, and workspace limits affect motion behavior.

Run:

    python singularity_demo2.py

### 2. Position-Assisted Pipe Welding Demo

**File:** `pipe_weld_rotating.py`

Demonstrates a simplified position-assisted pipe welding setup:

- pipe rotates
- torch stays mostly fixed
- robot joint speeds stay low
- simulation logs pipe angle and joint speed over time

Run:

    python pipe_weld_rotating.py

Controls:

    Space = start / pause
    R     = reset
    Mouse = orbit / zoom / pan

Example console output:

    Running: True
    step=0000 pipe_angle=   0.9 deg joint_speed= 0.163 rad/s
    step=0020 pipe_angle=  18.9 deg joint_speed= 0.164 rad/s
    step=0040 pipe_angle=  36.9 deg joint_speed= 0.161 rad/s
    step=0060 pipe_angle=  54.9 deg joint_speed= 0.156 rad/s
    step=0080 pipe_angle=  72.9 deg joint_speed= 0.149 rad/s

## Technical Themes

- Robotics visualization
- Forward/inverse kinematics concepts
- Singularity behavior
- Position-assisted automation
- Welding-style motion planning
- Simulation state logging
- Engineering visualization in Python

## Repository Contents

    robotics-simulation-demos/
    ├── README.md
    ├── singularity_demo2.py
    └── pipe_weld_rotating.py

## Notes

These are educational simulation demos using simplified geometry and toy parameters. They are not intended to represent production robot programming, certified welding paths, or proprietary manufacturing systems.

The purpose of this repository is to demonstrate simulation thinking, visual debugging, robotics concepts, and engineering-style software experimentation.
EOF
