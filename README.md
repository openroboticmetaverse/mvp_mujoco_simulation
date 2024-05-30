<p align="center">
  <a href="https://www.openroboticmetaverse.org">
    <img alt="orom" src="https://raw.githubusercontent.com/openroboverse/knowledge-base/main/docs/assets/icon.png" width="100" />
  </a>
</p>
<h1 align="center">
  🤖 open robotic metaverse mvp - robotics platform 🌐
</h1>

## Overview 🔍

This branch serves as an introduction to MuJoCo to familiarise new devs with its concepts.

## Technology Stack 🛠️

- **Simulation**: Developed using the Mujoco physics engine

## Setup ⚙️

### Cloning the Repository

1. Clone the repo:
   ```bash
   git clone https://github.com/openroboticmetaverse/mvp_mujoco_simulation.git
   ```
2. Navigate to the project directory:

   ```bash
   cd mvp_mujoco_simulation
   ```

3. Check out the "tutorial" branch:
   ```bash
   git checkout tutorial
   ```

### Setting Up a Virtual Environment

2. Navigate to the project directory:

   ```bash
   cd mvp_mujoco_simulation
   ```

3. Create a virtual environment:

   ```bash
   python3 -m venv mvp_mujoco
   ```

4. Activate the virtual environment:

   ```bash
   source mvp_mujoco/bin/activate
   ```

### Installing Dependencies

5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Simulation Locally 💻

### Starting the Simulation

6. Run the simulation script:
   ```bash
   cd src
   ```
   ```bash
   python mujoco_simulation.py
   ```

You should see the MuJoCo simulation running and visualized locally.

## Next Steps

Check out the [webapp](https://github.com/openroboticmetaverse/mvp-webapp), which is the other part of our mvp. To see the simulation running in your browser, follow the instructions and start the frontend (there is currently no need to run the backend).

## Acknowledgements

Kinematic calculations are taken from [Kevin Zakka](https://github.com/kevinzakka/mjctrl/) and the robot models are taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
