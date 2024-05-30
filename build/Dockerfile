# syntax=docker/dockerfile:1
FROM python:3.11-slim AS msdpostprocess_build

COPY *.* /
COPY src/MSDpostprocess/ /src/MSDpostprocess/
COPY build/build_data /build_data/
COPY build/QE_Pro_model/ /QE_Pro_Model/
COPY build/TOF_model/ /TOF_model/

#set up python environment with virtualenv so it can be copied to the second stage
RUN pip install virtualenv
RUN virtualenv --copies /env
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
RUN pip install -e .

#second stage docker container that doesn't have the training data
FROM python:3.11-slim

COPY --from=0 /env/ /env/
#this allows us to use the env without activating it
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

RUN mkdir models
COPY --from=0 /QE_Pro_model/ /models/QE_Pro_model/
COPY --from=0 /TOF_model/ /models/TOF_model/
#COPY --from=0 /MSDpostprocess/ /MSDpostprocess/
