# syntax=docker/dockerfile:1
FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get -y install python3-pip
RUN pip3 install dill==0.3 matplotlib==3.7 numpy==1.25 pandas==1.5 scikit-learn==1.2 scipy==1.10 statsmodels==0.13.5 brain-isotopic-distribution==1.5.11

COPY *.py /
COPY Asp37_Annotated_Lipids_for_steven.tsv /
#COPY model.dill /

RUN python3 SLT05_msDialPostprocess-trainingScript1.py -i /Asp37_Annotated_Lipids_for_steven.tsv -r 7 -o /
RUN rm -r QC_information
