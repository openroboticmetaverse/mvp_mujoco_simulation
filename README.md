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
docker exec -it mvp_backend bash
```
Go to src folder:
```bash
cd src
```
And run server
```bash
python3 mujoco_backend.py
```

## Run Test Client
Open a new terminal and open bash of container:
```bash
docker exec -it mvp_backend bash
```
Go to src folder:
```bash
cd src
```
And run server
```bash
python3 test_client.py
```


## Acknowledgements

Robot models are taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

