import json
from time import time
from typing import List

import tensorflow as tf
from tensorflow import keras

from src.analysis.accuracy_vs_kb import analyze_accuracy_vs_kb
from src.analysis.dataset import dataset_kb
from src.analysis.utils import release_models
from src.modelconfig import ModelConfig

compile_kwargs = {
    "loss": "sparse_categorical_crossentropy",
    "optimizer": keras.optimizers.RMSprop(),
    "metrics": ["accuracy"],
}


def model_by_name(model_name: str) -> keras.Model:
    model_path = f"models/{model_name}/{model_name}-full.h5"
    model = keras.models.load_model(model_path)
    model.compile(**compile_kwargs)
    return model


def analyze_latency(
    model: keras.Model, model_name: str, model_configs: List[ModelConfig]
):
    # TODO plot/tabulate latencies for each layer, and cumulative latency distribution
    # TODO also plot model layer sizes...? (transmitted over network)
    # TODO ^ also, plot for postencoders at various minimal acceptable accuracy % degradations, using accuracy_vs_kb analysis
    # TODO should be able to "estimate" or "simulate" total latency before deploying to mobile. :)

    dataset = dataset_kb()
    n = len(list(dataset))

    t1 = time()
    for img, _ in dataset.batch(1):
        _ = model.predict(img)
    t2 = time()
    ms = int(1000 * (t2 - t1) / n)
    print(f"{model_name}: {ms} ms")


def analyze_model(model_name: str, model_configs: List[ModelConfig]):
    model = model_by_name(model_name)

    analyze_latency(model, model_name, model_configs)

    # TODO jpeg only at the moment
    analyze_accuracy_vs_kb(model, model_configs)

    # TODO analyze_neuron_histogram
    # TODO analyze_video (accuracies? what are we analyzing here?)
    # TODO analyze_featuremap (why? just a visual?)
    # analyze sensitivity, static/dynamic components, etc

    release_models(model)


def main():
    with open("models.json") as f:
        d = json.load(f)

    model_configs = {
        model_name: [
            ModelConfig(model=model_name, **config_dict)
            for config_dict in config_dicts
        ]
        for model_name, config_dicts in d.items()
    }

    for model_name, model_configs_ in model_configs.items():
        analyze_model(model_name, model_configs_)


if __name__ == "__main__":
    main()


# TODO move generate_models analysis stuff here...
