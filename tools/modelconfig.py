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

    def to_path(self) -> str:
        enc = ModelConfig._encdec_str(self.encoder, self.encoder_args)
        dec = ModelConfig._encdec_str(self.decoder, self.decoder_args)
        return f"{self.model}/{self.model}-{self.layer}-{enc}-{dec}"

    @staticmethod
    def _encdec_str(name, args):
        if len(args) == 0:
            return name
        return f"{name}({_dict_to_str(args)})"


def _dict_to_str(d: dict, sep: str = ", ") -> str:
    return sep.join(f"{k}={v}" for k, v in sorted(d.items()))
