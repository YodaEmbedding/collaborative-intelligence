import json
from typing import List, Tuple

from src.modelconfig import ModelConfig


def json_confirmation(frame_number: int, num_bytes: int) -> str:
    return json.dumps(
        {
            "type": "confirmation",
            "frameNumber": frame_number,
            "numBytes": num_bytes,
        }
    )


def json_ping(id_) -> str:
    return json.dumps({"type": "ping", "id": id_})


def json_ready(model_config: ModelConfig) -> str:
    return json.dumps(
        {"type": "ready", "modelConfig": model_config.to_json_object()}
    )


def json_result(
    frame_number: int,
    # read_time: int,
    # feed_time: int,
    inference_time: int,
    predictions: List[Tuple[str, str, float]],
) -> str:
    return json.dumps(
        {
            "type": "result",
            "frameNumber": frame_number,
            # "readTime": read_time,  # TODO?
            # "feedTime": feed_time,
            "inferenceTime": inference_time,
            "predictions": [
                {"name": name, "description": desc, "score": score}
                for name, desc, score in predictions
            ],
        }
    )
