# Flask settings
DEFAULT_FLASK_SERVER_NAME = '0.0.0.0'
DEFAULT_FLASK_SERVER_PORT = '5000'
DEFAULT_FLASK_DEBUG = True  # Do not use debug mode in production

# Flask-Restplus settings
RESTPLUS_SWAGGER_UI_DOC_EXPANSION = 'list'
RESTPLUS_VALIDATE = True
RESTPLUS_MASK_SWAGGER = False
RESTPLUS_ERROR_404_HELP = False

# AD client settings
DEFAULT_TF_SERVER_NAME = 'tensorflow_serving'
DEFAULT_TF_SERVER_PORT = 9000
AD_MODEL_NAME = 'model'
AD_MODEL_SIGNATURE_NAME = 'predict_ads'
AD_MODEL_INPUTS_KEY = 'images'
