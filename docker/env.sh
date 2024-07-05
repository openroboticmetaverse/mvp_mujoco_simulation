#!/bin/bash
set -e

echo "Creating Python virtual environment"
python3 -m venv myEnv

echo "Installing python packages"
./myEnv/bin/pip install -r /sim_ws/requirements.txt

echo "Start simulation"
./myEnv/bin/python3 /sim_ws/src/mujoco_simulation.py