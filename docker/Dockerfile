FROM ubuntu:latest  
ARG WS_PATH=/sim_ws
SHELL ["/bin/bash", "-c", "-o", "pipefail"]

RUN apt-get update
RUN apt-get install -y x11-apps x11-utils libx11-dev  python3-venv\
                        python3-pip \
                        git

RUN mkdir -p $WS_PATH
WORKDIR $WS_PATH/
ENV WS_PATH $WS_PATH

COPY ./src ./src
COPY ./config ./config
COPY ./requirements.txt ./requirements.txt

COPY ./docker/env.sh ./start.sh
RUN chmod +x start.sh

RUN python3 -m venv myEnv
ENV PATH "$PWD/myEnv/bin:$PATH"

RUN myEnv/bin/python3 -m pip install -r requirements.txt

RUN echo "source $WS_PATH/start.sh" >> /root/.bashrc

ENV DISPLAY "unix:0"

CMD ["/bin/bash"]
