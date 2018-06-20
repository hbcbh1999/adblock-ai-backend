FROM ubuntu:16.04

RUN apt-get update -y
RUN apt-get update -y && apt-get install -y \
    build-essential \
    python \
    python-pip \
    libopencv-dev \
    python-opencv

RUN pip install --upgrade pip

COPY . /facebook-bot
WORKDIR /facebook-bot

ENV VERIFY_TOKEN="ADBLOCKPLUS_FTW"

RUN pip install -r requirements.txt

EXPOSE 5000

#CMD ["python ./app/app.py"]