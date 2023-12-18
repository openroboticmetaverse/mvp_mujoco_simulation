# MuJoCo Controllers

Single-file pedagogical implementations of common robotics controllers for MuJoCo.

## Installation

MuJoCo is the only dependency required to run the controllers.

```bash
pip install "mujoco>=3.1.0"
```

## Controllers

* [differential inverse kinematics using the pseudoinverse](diffik.py)
* differential inverse kinematics using quadratic programming
  * [diffik_qp](diffik_qp.py): mujoco-only implementation, uses `mju_boxQP`.
  * [diffik_qpsolvers](diffik_qpsolvers.py): uses `qpsolvers` for a comparison to the mujoco-only implementation.

## Acknowledgements

Robot models are taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
