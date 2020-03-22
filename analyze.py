import json
from time import time
from typing import List

import numpy as np
from tensorflow import keras

from src.analysis.accuracy_vs_kb import analyze_accuracy_vs_kb
from src.analysis.dataset import dataset_kb
from src.analysis.utils import prefix_of, release_models
from src.modelconfig import ModelConfig
from src.utils import split_model_by_config

compile_kwargs = {
    "loss": "sparse_categorical_crossentropy",
    "optimizer": keras.optimizers.RMSprop(),
    "metrics": [],
    "run_eagerly": False,
}


def model_by_name(model_name: str) -> keras.Model:
    model_path = f"models/{model_name}/{model_name}-full.h5"
    model = keras.models.load_model(model_path, compile=False)
    return model


def analyze_latency(
    model: keras.Model, model_name: str, model_configs: List[ModelConfig]
):
    # TODO plot/tabulate latencies for each layer, and cumulative latency distribution
    # TODO also plot model layer sizes...? (transmitted over network)
    # TODO ^ also, plot for postencoders at various minimal acceptable accuracy % degradations, using accuracy_vs_kb analysis
    # TODO should be able to "estimate" or "simulate" total latency before deploying to mobile. :)

    BATCH_SIZE = 1
    dataset = dataset_kb()
    n = len(list(dataset))
    model.compile(**compile_kwargs)
    t1 = time()

    # TODO has further perf improvements in TF 2.1? so maybe upgrade ubuntu
    for img, _ in dataset.batch(1):
        _ = model.predict_on_batch(img)

    t2 = time()
    ms = int(1000 * (t2 - t1) / n)
    print(f"{model_name}: {ms} ms")

    model.compile(**compile_kwargs)
    t1 = time()
    model.evaluate(dataset.batch(BATCH_SIZE), verbose=0)
    t2 = time()
    ms = int(1000 * (t2 - t1) / n)
    print(f"{model_name}: {ms} ms")


def analyze_distribution(model: keras.Model, model_configs: List[ModelConfig]):
    BATCH_SIZE = 64

    model_configs = [
        x for x in model_configs if x.layer != "server" and x.layer != "client"
    ]
    dataset = dataset_kb()

    def floatlike(x):
        try:
            float(x)
            return True
        except:
            return False

    for model_config in model_configs:
        print(f"Analyzing distribution\n{model_config}\n")
        prefix = prefix_of(model_config)
        model_client, model_server, model_analysis = split_model_by_config(
            model, model_config
        )

        pred_analysis = model_analysis.predict(dataset.batch(BATCH_SIZE))
        pred = {
            "split": pred_analysis[0],
            "final": pred_analysis[-1],
        }
        if len(pred_analysis) == 4:
            pred["encoded"] = pred_analysis[1]
            pred["decoded"] = pred_analysis[2]

        for split, p in pred.items():
            stats = {
                "type": split,
                "shape": p.shape,
                "mean": np.mean(p),
                "std": np.std(p),
                "min": np.min(p),
                "max": np.max(p),
                "pct 0.1": np.percentile(p, 0.1),
                "pct 1": np.percentile(p, 1),
                "pct 5": np.percentile(p, 5),
                "pct 95": np.percentile(p, 95),
                "pct 99": np.percentile(p, 99),
                "pct 99.9": np.percentile(p, 99.9),
            }

            print([type(x) for x in stats.values()])
            stats = {
                k: f"{v:.3g}" if floatlike(v) else v for k, v in stats.items()
            }
            print("\n".join(f"{k}: {v}" for k, v in stats.items()))
            print("")

        # TODO print max, min, mean, stddev
        # max/min also have their own mean/stddev
        # There's also min/max of the means... and stddev of the means...

        # TODO _plot(prefix, ...)

        release_models(model_client, model_server, model_analysis)


def analyze_model(model_name: str, model_configs: List[ModelConfig]):
    model = model_by_name(model_name)

    # analyze_distribution(model, model_configs)
    # analyze_latency(model, model_name, model_configs)
    analyze_accuracy_vs_kb(model, model_configs)
    # TODO jpeg only at the moment
    # TODO analyze_neuron_histogram
    # TODO analyze_video (accuracies? what are we analyzing here?)
    # TODO analyze_featuremap (why? just a visual?)
    # analyze sensitivity, static/dynamic components, etc

    release_models(model)


# TODO Refactor

# ModelConfig based:
# For each model,
#   For each split by model_config,
#       U8: Determine good range via mean/std/etc
#       ...
#       layer output histogram
#       baseline accuracy (simple? no compress?)

#       accuracy_vs_kb...???

# Not ModelConfig based:
# For each model,
#   For each split,
#       latency???? (do we really need to split for this? I guess... but we should )


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
# TODO include scripts for generating data/{}kb
