import json
from typing import List

from tensorflow import keras

from src.analysis.accuracy_vs_kb import analyze_accuracy_vs_kb
from src.analysis.utils import release_models
from src.modelconfig import ModelConfig

compile_kwargs = {
    "loss": "sparse_categorical_crossentropy",
    "optimizer": keras.optimizers.RMSprop(),
    "metrics": ["accuracy"],
}


def analyze_model(model_name: str, model_configs: List[ModelConfig]):
    model_path = f"models/{model_name}/{model_name}-full.h5"
    model: keras.Model = keras.models.load_model(model_path)
    model.compile(**compile_kwargs)
    analyze_accuracy_vs_kb(model, model_configs)
    # TODO jpeg only at the moment
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
