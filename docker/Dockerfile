FROM ubuntu:latest  
ARG WS_PATH=/sim_ws
SHELL ["/bin/bash", "-c", "-o", "pipefail"]

RUN apt-get update
RUN apt-get install -y x11-apps x11-utils libx11-dev \
                        python3-pip \
                        git

RUN python3 -m pip install  mujoco \
                            websockets \
                            numpy \
                            mediapy\
                            # temporary
                            tqdm \
                            ffmpeg

RUN mkdir -p $WS_PATH
WORKDIR $WS_PATH/
COPY ./src ./src
COPY ./config ./config
WORKDIR $WS_PATH

ENV DISPLAY "unix:0"
ENV WS_PATH $WS_PATH
CMD ["/bin/bash"]

