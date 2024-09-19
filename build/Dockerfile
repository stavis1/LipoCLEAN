# syntax=docker/dockerfile:1
FROM python:3.11.9-slim AS lipoclean_build

#add the files necessary to install lipoCLEAN
RUN mkdir src
COPY src/lipoCLEAN/ /src/lipoCLEAN/
COPY *.* /

#add the trained models
RUN mkdir models
COPY build/QE_Pro_model/ /models/QE_Pro_model/
COPY build/TOF_model/ /models/TOF_model/

#set up python environment with virtualenv
COPY build/requirements.txt /
RUN pip install -r requirements.txt
RUN pip install .

#we want --print to return a different default options for the docker version
#so the tool checks this environment variable
ENV AM_I_IN_A_DOCKER_CONTAINER=Yes

