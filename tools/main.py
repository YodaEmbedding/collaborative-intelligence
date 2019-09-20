import errno
import os
import warnings
from pprint import pprint
from typing import Callable, Tuple

# TODO
# import tensorflow.keras
# from tensorflow.keras.layers import Layer

warnings.filterwarnings('ignore', category=FutureWarning)
from classification_models.tfkeras import Classifiers
import numpy as np
import tensorflow as tf
from tensorflow import keras
# pylint: disable-msg=E0611
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import plot_model
#pylint: enable-msg=E0611

from split import EncoderLayer, DecoderLayer, split_model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def convert_to_tflite_model(
    keras_model_filename: str,
    tflite_filename: str,
    **kwargs
):
    """Convert keras model file to tflite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model_file(
        keras_model_filename, **kwargs)
    tflite_model = converter.convert()
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)

def create_directory(directory: str):
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def create_model(model_name: str) -> keras.Model:
    """Model factory."""
    shape = (224, 224, 3)
    model_creator, _ = Classifiers.get(model_name)
    return model_creator(shape, weights='imagenet')

def get_preprocessor(model_name: str):
    """Get input preprocessor for model."""
    return Classifiers.get(model_name)[1]

def get_encoder_decoder(
    model_name: str
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Get encoder/decoder for model."""
    if model_name.startswith('resnet'):
        clip_range = (-2, 2)
        return EncoderLayer(clip_range), DecoderLayer(clip_range)
    identity = lambda x: x
    return identity, identity

def single_input_image(filename: str):
    """Load single image for testing."""
    img = image.load_img(filename, target_size=(224, 224))
    imgs = image.img_to_array(img)
    imgs = np.expand_dims(imgs, axis=0)
    return imgs

def write_summary_to_file(model: keras.Model, filename: str):
    with open(filename, 'w') as f:
        model.summary(print_fn=lambda x: f.write(f'{x}\n'))

def main():
    model_name = 'vgg19'
    prefix = f'{model_name}/{model_name}'
    split_layer_name = {
        'resnet18':  'add_5',   # (14, 14, 256)
        # 'resnet18':  'add_7', # (7, 7, 512)
        'resnet34':  'add_8',   # (14, 14, 256)
        'resnet50':  'add_8',   # (14, 14, 1024)
        'resnet101': 'add_8',   # (14, 14, 1024)
        'resnet152': 'add_12',  # (14, 14, 1024)
        'vgg19': 'block4_pool',  # (14, 14, 512)
    }[model_name]
    prefix_split = f'{model_name}/{model_name}-{split_layer_name}'

    preprocess_input = get_preprocessor(model_name)
    test_images = preprocess_input(single_input_image('sample.jpg'))

    # Load and save entire model
    try:
        model = keras.models.load_model(f'{prefix}-full.h5')
    except OSError:
        model = create_model(model_name)
        create_directory(model_name)
        plot_model(model, to_file=f'{prefix}-full.png')
        model.save(f'{prefix}-full.h5')

    # Force usage of tf.keras.Model, which appears to have Nodes linked correctly
    model = keras.models.load_model(f'{prefix}-full.h5')

    # Make predictions
    predictions = model.predict(test_images)
    predictions = imagenet_utils.decode_predictions(predictions)
    write_summary_to_file(model, f'{prefix}-full.txt')
    pprint(predictions)

    # Load and save split model
    try:
        # TODO Don't really need custom_objects for all models...
        model_client = keras.models.load_model(
            f'{prefix_split}-client.h5',
            custom_objects={'EncoderLayer': EncoderLayer})
        model_server = keras.models.load_model(
            f'{prefix_split}-server.h5',
            custom_objects={'DecoderLayer': DecoderLayer})
    except OSError:
        encoder, decoder = get_encoder_decoder(model_name)
        model_client, model_server = split_model(
            model, split_layer_name, encoder, decoder)
        model_client.save(f'{prefix_split}-client.h5')
        model_server.save(f'{prefix_split}-server.h5')
        plot_model(model_client, to_file=f'{prefix_split}-client.png')
        plot_model(model_server, to_file=f'{prefix_split}-server.png')

    # Make predictions
    prev_predictions = predictions
    predictions = model_client.predict(test_images)
    predictions = model_server.predict(predictions)
    predictions = imagenet_utils.decode_predictions(predictions)
    write_summary_to_file(model, f'{prefix_split}-client.txt')
    write_summary_to_file(model, f'{prefix_split}-server.txt')
    pprint(predictions)
    print('Same predictions with split model? '
          f'{np.all(predictions == prev_predictions)}')

    # Save TFLite models
    convert_to_tflite_model(
        f'{prefix}-full.h5',
        f'{prefix}-full.tflite')

    convert_to_tflite_model(
        f'{prefix_split}-client.h5',
        f'{prefix_split}-client.tflite',
        custom_objects={'EncoderLayer': EncoderLayer})

if __name__ == "__main__":
    main()
