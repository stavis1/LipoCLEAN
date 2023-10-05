# syntax=docker/dockerfile:1
FROM python:3.9-slim

RUN pip install dill==0.3 matplotlib==3.7 numpy==1.25 pandas==1.5 scikit-learn==1.2 scipy==1.10 statsmodels==0.13.5 brain-isotopic-distribution==1.5.11

COPY scripts/*.py /
COPY example_data/training_data.tsv /

RUN python3 MSDpostprocess-training.py -i /training_data.tsv -r 7 -n -o /
RUN chmod +x MSDpostprocess-training.py
RUN chmod +x MSDpostprocess-inference.py
ENV PATH="${PATH}:/"

RUN mkdir data
