# syntax=docker/dockerfile:1
FROM python:3.11-slim AS msdpostprocess_build

COPY environments/requirements.txt /
COPY src/MSDpostprocess/ /MSDpostprocess/
COPY build/build_data /build_data/

#set up python environment with virtualenv so it can be copied to the second stage
RUN pip install virtualenv
RUN virtualenv --copies /env
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
RUN pip install -r requirements.txt

#train the models
RUN python /MSDpostprocess --options build_data/train_QE_Pro_model.toml
RUN python /MSDpostprocess --options build_data/train_TOF_model.toml

#second stage docker container that doesn't have the training data
FROM python:3.11-slim

COPY --from=0 /env/ /env/
#this allows us to use the env without activating it
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

RUN mkdir models
COPY --from=0 /QE_Pro_model/ /models/QE_Pro_model/
COPY --from=0 /TOF_model/ /models/TOF_model/
COPY --from=0 /MSDpostprocess/ /MSDpostprocess/
