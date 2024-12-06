FROM python:slim-bookworm

SHELL ["/bin/bash", "-c", "-o", "pipefail"]

RUN apt-get update
RUN apt-get install -y x11-apps x11-utils libx11-dev \
                        git

COPY ./src /sim_ws/src
COPY ./requirements.txt /sim_ws/requirements.txt
WORKDIR /sim_ws

RUN pip install -r /sim_ws/requirements.txt
