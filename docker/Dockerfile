FROM ubuntu:latest  

SHELL ["/bin/bash", "-c", "-o", "pipefail"]

RUN apt-get update && apt-get upgrade
RUN apt-get install -y x11-apps x11-utils libx11-dev python3-venv\
                        python3-pip \
                        git

COPY ./requirements.txt /sim_ws/requirements.txt
COPY ./docker/env.sh /sim_ws/start.sh
RUN chmod +x /sim_ws/start.sh

WORKDIR /sim_ws
RUN sed -i 's/\r$//' /sim_ws/start.sh
ENTRYPOINT ["/bin/bash", "/sim_ws/start.sh"]
