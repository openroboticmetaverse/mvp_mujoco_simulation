
<p align="center">
  <a href="https://www.openroboticmetaverse.org">
    <img alt="orom" src="https://raw.githubusercontent.com/openroboverse/knowledge-base/main/docs/assets/icon.png" width="100" />
  </a>
</p>
<h1 align="center">
  🤖 Open Robotic Metaverse MVP - Robotics Platform 🌐
</h1>


## Overview 🔍

This project serves as the MVP (Minimum Viable Product) 🚀 for a larger vision aimed at developing a robotic metaverse. Utilizing a combination of modern web technologies, this platform allows users to interact with robots through a web browser, fostering a unique and interactive environment.

## Technology Stack 🛠️

- **Simulation**: Developed using the Mujoco physics engine


## Setup ⚙️

1. Clone the repo:
```bash
git clone https://github.com/openroboticmetaverse/mvp_mujoco_simulation.git
```

2. Run container:
```bash
cd mvp_mujoco_simulation
```
```bash
docker compose up -d
```

## Start the Simulation 💻

Open a console in the container:
```bash
docker exec -it mvp_simulation bash
```

Start the simulation:
```bash
cd src
```
```bash
python3 mujoco_simulation.py
```

## Test the Simulation 💻

Open a console in the container:
```bash
docker exec -it mvp_simulation bash
```

Start a test client to see if the simulation websocket is working:
```bash
cd src
```
```bash
python3 test_client.py
```
You should be able to see the datastream printed out in the console, in which you executed the _test_client.py_. After a short time the script stops, now you should see the message "Connection closed - OK" in the console of the simulation.


## Next Steps
Check out the [webapp](https://github.com/openroboticmetaverse/mvp-webapp), which is the other part of our mvp. To see the simulation running in your browser, follow the instructions and start the frontend (there is currently no need to run the backend).



## Acknowledgements
Kinematic calculations are taken from [Kevin Zakka](https://github.com/kevinzakka/mjctrl/) and the robot models are taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
