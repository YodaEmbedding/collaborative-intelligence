from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(eq=True, frozen=True)
class ModelConfig:
    model: str
    layer: str
    encoder: str = "None"
    decoder: str = "None"
    encoder_args: Dict[str, Any] = field(default_factory=lambda: {})
    decoder_args: Dict[str, Any] = field(default_factory=lambda: {})

    def __hash__(self) -> int:
        return hash(json.dumps(asdict(self)))

    def to_json_object(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        if len(self.encoder_args) == 0:
            d.pop("encoder_args")
        if len(self.decoder_args) == 0:
            d.pop("decoder_args")
        return d

    def to_path(self) -> str:
        enc = ModelConfig._encdec_str(self.encoder, self.encoder_args)
        dec = ModelConfig._encdec_str(self.decoder, self.decoder_args)
        return "&".join((f"{self.model}/{self.model}", self.layer, enc, dec))

    @staticmethod
    def _encdec_str(name, args) -> str:
        if len(args) == 0:
            return name
        return f"{name}({ModelConfig._dict_to_str(args)})"

    @staticmethod
    def _dict_to_str(d: dict, sep: str = ",") -> str:
        return sep.join(
            f"{k}={ModelConfig._stringify(v)}" for k, v in sorted(d.items())
        )

    @staticmethod
    def _stringify(x):
        return json.dumps(x, separators=(",", "="))

    @staticmethod
    def from_json_dict(d: Dict[str, Any]) -> ModelConfig:
        d = {k: v for k, v in d.items() if v is not None}
        return ModelConfig(**d)


@dataclass(eq=True, frozen=True)
class PostencoderConfig:
    type: str
    quality: int

    @staticmethod
    def from_json_dict(d: Dict[str, Any]) -> PostencoderConfig:
        return PostencoderConfig(**d)


@dataclass(eq=True, frozen=True)
class ProcessorConfig:
    model_config: ModelConfig
    postencoder_config: PostencoderConfig

    @staticmethod
    def from_json_dict(d: Dict[str, Any]) -> ProcessorConfig:
        return ProcessorConfig(
            ModelConfig.from_json_dict(d["model_config"]),
            PostencoderConfig.from_json_dict(d["postencoder_config"]),
        )
