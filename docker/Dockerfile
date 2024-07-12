FROM python:slim-bookworm

SHELL ["/bin/bash", "-c", "-o", "pipefail"]

RUN apt-get update && apt-get upgrade
RUN apt-get install -y x11-apps x11-utils libx11-dev \
                        git

COPY ./requirements.txt /sim_ws/requirements.txt
WORKDIR /sim_ws

RUN pip install -r /sim_ws/requirements.txt \
    python3 /sim_ws/src/mujoco_simulation.py

