# NOTE: This is based on a comment here: https://github.com/tensorflow/serving/issues/819
FROM ubuntu:16.04

# Install general packages
RUN apt-get update && apt-get install -y \
        curl \
        libcurl3-dev \
        unzip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Previous Installation of tensorflow-model-server (BROKEN RECENTLY)
#RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list \
#    && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - \
#    && apt-get update && apt-get install tensorflow-model-server

# New installation of tensorflow-model-server
RUN TEMP_DEB="$(mktemp)" \
    && wget -O "$TEMP_DEB" 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-1.5.0/t/tensorflow-model-server/tensorflow-model-server_1.5.0_all.deb' \
    && dpkg -i "$TEMP_DEB" \
    && rm -f "$TEMP_DEB" 

RUN mkdir /var/fb-model
ADD model /var/fb-model

EXPOSE 9000

# Serve the model when the container starts
ENTRYPOINT ["tensorflow_model_server"]
CMD ["--port=9000", "--model_name=model", "--model_base_path=/var/fb-model"]