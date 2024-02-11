# MuJoCo Backend Server

MVP Version of our backend using MuJoCo

## Start Server

```bash
docker compose build
```
```bash
docker compose up
```

Open a new terminal and open bash of container:
```bash
docker exec -it mvp_simulation bash
```
Go to src folder:
```bash
cd src
```
And run server
```bash
python3 mujoco_simulation.py
```

## Run Test Client
Open a new terminal and open bash of container:
```bash
docker exec -it mvp_simulation bash
```
Go to src folder:
```bash
cd src
```
And run server
```bash
python3 test_client.py
```

## Debugging
In case you get the error: 
```bash
Authorization required, but no authorization protocol specified
/usr/local/lib/python3.10/dist-packages/glfw/__init__.py:916: GLFWError: (65544) b'X11: Failed to open display :0'
  warnings.warn(message, GLFWError)
ERROR: could not initialize GLFW
```
Try to run ```xhost +``` before executing the server.
Also check if your display variable is set correctly inside the container by running ```echo $DISPLAY```

## Acknowledgements

Robot models are taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

