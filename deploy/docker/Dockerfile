#!/bin/bash
FROM continuumio/miniconda3:latest as builder
ARG USERNAME=mle-dhikshita
ARG USER_UID=1000
ARG USER_GID=$USER_UID
COPY . /mle-training/
WORKDIR /mle-training/
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 777 /etc/sudoers.d/$USERNAME
RUN chown -R $USERNAME:$USER_GID /mle-training/
COPY ./deploy/conda/env.yml .
RUN conda env create -f env.yml
RUN echo "source activate mle-dev" > ~/.bashrc
ENV PATH /opt/conda/envs/mle-dev/bin:$PATH
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get install -y python3-pip && \
    apt-get install -y python3
RUN python3 -m build && \
    pip install dist/*.whl --force-reinstall

FROM builder as builder1
WORKDIR /mle-training/
ENV PATH /opt/conda/envs/mle-dev/bin:$PATH
ENV MLFLOW_TRACKING_URI=http://localhost:5008
EXPOSE 5008
#COPY ./deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 777 ./deploy/docker/entrypoint.sh
RUN chown $USERNAME:$USER_GID ./deploy/docker/entrypoint.sh
USER $USERNAME
ENTRYPOINT ["/bin/bash", "./deploy/docker/entrypoint.sh"]