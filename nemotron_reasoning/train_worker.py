from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nemotron_reasoning.training import train_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Torchrun worker for Modal LoRA training.")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    config: dict[str, Any] = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    train_lora_adapter(**config)


if __name__ == "__main__":
    main()
