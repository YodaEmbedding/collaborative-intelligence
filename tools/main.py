import errno
import os
from pprint import pprint

from classification_models.resnet import (
    preprocess_input, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
import numpy as np
import tensorflow as tf
from tensorflow import keras
# pylint: disable-msg=E0611
from tensorflow.python.keras.applications import imagenet_utils
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
    d = {
        'resnet18':  lambda: ResNet18(input_shape=shape, weights='imagenet'),
        'resnet34':  lambda: ResNet34(input_shape=shape, weights='imagenet'),
        'resnet50':  lambda: ResNet50(input_shape=shape, weights='imagenet'),
        'resnet101': lambda: ResNet101(input_shape=shape, weights='imagenet'),
        'resnet152': lambda: ResNet152(input_shape=shape, weights='imagenet'),
    }
    return d[model_name]()

def single_input_image(filename: str):
    """Load single image for testing."""
    img = image.load_img(filename, target_size=(224, 224))
    imgs = image.img_to_array(img)
    imgs = np.expand_dims(imgs, axis=0)
    return preprocess_input(imgs)

def write_summary_to_file(model: keras.Model, filename: str):
    with open(filename, 'w') as f:
        model.summary(print_fn=lambda x: f.write(f'{x}\n'))

def main():
    model_name = 'resnet34'
    prefix = f'{model_name}/{model_name}'
    split_layer_name = {
        'resnet18':  'add_5',   # (14, 14, 256)
        # 'resnet18':  'add_7', # (7, 7, 512)
        'resnet34':  'add_8',   # (14, 14, 256)
        'resnet50':  'add_8',   # (14, 14, 1024)
        'resnet101': 'add_8',   # (14, 14, 1024)
        'resnet152': 'add_12',  # (14, 14, 1024)
    }[model_name]

    test_images = single_input_image('sample.jpg')

    # Load and save entire model
    try:
        model = keras.models.load_model(f'{prefix}-full.h5')
    except OSError:
        model = create_model(model_name)
        create_directory(model_name)
        # model.load_weights(f'{prefix}_imagenet_1000.h5')  # Unnecessary
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
        model_client = keras.models.load_model(
            f'{prefix}-client.h5',
            custom_objects={'EncoderLayer': EncoderLayer})
        model_server = keras.models.load_model(
            f'{prefix}-server.h5',
            custom_objects={'DecoderLayer': DecoderLayer})
    except OSError:
        model_client, model_server = split_model(model, split_layer_name)
        model_client.save(f'{prefix}-client.h5')
        model_server.save(f'{prefix}-server.h5')
        plot_model(model_client, to_file=f'{prefix}-client.png')
        plot_model(model_server, to_file=f'{prefix}-server.png')

    # Make predictions
    prev_predictions = predictions
    predictions = model_client.predict(test_images)
    predictions = model_server.predict(predictions)
    predictions = imagenet_utils.decode_predictions(predictions)
    write_summary_to_file(model, f'{prefix}-client.txt')
    write_summary_to_file(model, f'{prefix}-server.txt')
    pprint(predictions)
    print('Same predictions with split model? '
          f'{np.all(predictions == prev_predictions)}')

    # Save TFLite models
    convert_to_tflite_model(
        f'{prefix}-full.h5',
        f'{prefix}-full.tflite')

    convert_to_tflite_model(
        f'{prefix}-client.h5',
        f'{prefix}-client.tflite',
        custom_objects={'EncoderLayer': EncoderLayer})

if __name__ == "__main__":
    main()
