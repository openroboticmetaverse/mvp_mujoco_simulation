<p align="center">
  <a href="https://www.openroboticmetaverse.org">
    <img alt="orom" src="https://raw.githubusercontent.com/openroboverse/knowledge-base/main/docs/assets/icon.png" width="100" />
  </a>
</p>
<h1 align="center">
  🤖 open robotic metaverse mvp - robotics platform 🌐
</h1>

## Overview 🔍

This branch serves as preliminary solution for generating an xml file combining multiple models using scene coordinates. For now, the script is only able to combine models of the same type.

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

3. Change branch:

   ```bash
   git switch mutli-model_simulation
   ```

### Setting Up Docker Container

2. Navigate to the project directory:

   ```bash
   docker compose up -d   
   ```
   ```bash
   docker exec -it mvp_simulation bash   
   ```
   
### Start Simulation
   ```bash
    cd src
   ```
   ```bash
    python3 multi-model_simulation.py --robot <robot_name> -n <number_of_robots>
   ```
   Replace `<robot_name>` with one of the supported robot models.
   Replace `<robot_number>` with the number of robots you want shown in the simulation
   Supported Robot Models: `kuka`, `franka`, `ur5e`, `anymal_b`, `anymal_c`, `cf2`, `boston_d`, `fruitfly`, `aloha`, `cassie`, `franka3`, `google_robot`
You should see the MuJoCo simulation running and visualized locally.

## Next Steps

The next step involves expanding the current solution to host multiple types of robots and objects at the same time.

## Acknowledgements

Kinematic calculations are taken from [Kevin Zakka](https://github.com/kevinzakka/mjctrl/) and the robot models are taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
