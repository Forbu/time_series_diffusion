# import torch docker image
FROM pytorch/pytorch:latest

RUN pip3 install pandas numpy pytest
 
RUN apt update 
RUN apt install curl
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN export PATH="/root/.local/bin:$PATH"

RUN pip3 install lightning