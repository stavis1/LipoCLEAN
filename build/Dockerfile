# syntax=docker/dockerfile:1
FROM python:3.11.9-slim AS lipoclean_build

#add the files necessary to install lipoCLEAN
RUN mkdir src
COPY src/lipoCLEAN/ /src/lipoCLEAN/
COPY *.* /

#add the trained models
RUN mkdir models
COPY build/MSD4_QE_Pro_model/ /models/MSD4_QE_Pro_model/
COPY build/MSD5_QE_Pro_model/ /models/MSD5_QE_Pro_model/
COPY build/MSD4_TOF_model/ /models/MSD4_TOF_model/

#set up python environment
RUN apt update -y && apt upgrade -y && apt install build-essential -y
COPY build/requirements.txt /
RUN pip install -r requirements.txt --default-timeout=100
RUN pip install .

#we want --print to return a different default options for the docker version
#so the tool checks this environment variable
ENV AM_I_IN_A_DOCKER_CONTAINER=Yes

