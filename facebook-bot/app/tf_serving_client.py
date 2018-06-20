from __future__ import print_function

import os
import operator
import logging
import settings
import utils

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import tensorflow as tf

log = logging.getLogger(__name__)

def __get_tf_server_connection_params__():
    '''
    Returns connection parameters to TensorFlow Server

    :return: Tuple of TF server name and server port
    '''
    server_name = utils.get_env_var_setting('TF_SERVER_NAME', settings.DEFAULT_TF_SERVER_NAME)
    server_port = utils.get_env_var_setting('TF_SERVER_PORT', settings.DEFAULT_TF_SERVER_PORT)

    return server_name, server_port

def __create_prediction_request__(input_params):
    '''
    Creates prediction request to TensorFlow server for AD model

    :param: Byte array, image for prediction
    :return: PredictRequest object
    '''
    # create predict request
    request = predict_pb2.PredictRequest()

    # Call AD model to make prediction on the image
    request.model_spec.name = "model"
    request.model_spec.signature_name = "predict"

    input_size = 608
    max_box_per_image = 4

    input_image_proto = tf.contrib.util.make_tensor_proto(input_params[0], shape=[1, input_size, input_size, 3], dtype=types_pb2.DT_FLOAT)
    true_boxes_proto = tf.contrib.util.make_tensor_proto(input_params[1], shape=[1,1,1,1,max_box_per_image,4], dtype=types_pb2.DT_FLOAT)
    
#    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dims=[input_size, input_size, 3])
#    input_image_proto = tensor_pb2.TensorProto(
#        dtype=types_pb2.DTcd _FLOAT,
#        tensor_shape=tensor_shape_proto,
#        float_val=[input_params[0]])
    
    request.inputs["input_image"].CopyFrom(input_image_proto)
    request.inputs["true_boxes"].CopyFrom(true_boxes_proto)

    return request

def __open_tf_server_channel__(server_name, server_port):
    '''
    Opens channel to TensorFlow server for requests

    :param server_name: String, server name (localhost, IP address)
    :param server_port: String, server port
    :return: Channel stub
    '''
    channel = implementations.insecure_channel(
        server_name,
        int(server_port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    return stub

def __make_prediction_and_prepare_results__(stub, request):
    '''
    Sends Predict request over a channel stub to TensorFlow server

    :param stub: Channel stub
    :param request: PredictRequest object
    :return: List of tuples, 3 most probable digits with their probabilities
    '''
    result = stub.Predict(request, 60.0)  # 60 secs timeout
    return tf.contrib.util.make_ndarray(result.outputs['outputs'])[0,:,:,:,:]

def make_prediction(image):
    '''
    Detect ads on a screenshot

    :param image: Byte array, images for prediction
    :return: image with ads highlighted
    '''
    # get TensorFlow server connection parameters
    server_name, server_port = __get_tf_server_connection_params__()
    log.info('Connecting to TensorFlow server %s:%s', server_name, server_port)

    # open channel to tensorflow server
    stub = __open_tf_server_channel__(server_name, server_port)

    # create predict request
    request = __create_prediction_request__(image)

    # make prediction
    return __make_prediction_and_prepare_results__(stub, request)
