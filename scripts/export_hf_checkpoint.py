import argparse
import json
import shutil
from pathlib import Path

import yaml
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from fogen.hf_model import FogenConfig, FogenForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training_config = yaml.safe_load(open(args.config))
    model_config = FogenConfig(**training_config["model"])
    model = FogenForCausalLM(model_config)
    state = {key: value.float() for key, value in load_file(args.ckpt).items()}
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(key.startswith("rope_") for key in missing)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.tokenizer_dir) / "tokenizer.json"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(output)

    # Copy model source files for trust_remote_code loading
    src_dir = Path(__file__).resolve().parents[1] / "src" / "fogen"
    shutil.copy2(src_dir / "hf_model.py", output / "hf_model.py")
    shutil.copy2(src_dir / "model.py", output / "model.py")
    # Patch import to be relative (for loading from model dir)
    hf_model_path = output / "hf_model.py"
    text = hf_model_path.read_text()
    hf_model_path.write_text(text.replace("from fogen.model import", "from .model import"))

    # Add auto_map to config.json
    config_path = output / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict["auto_map"] = {
        "AutoConfig": "hf_model.FogenConfig",
        "AutoModelForCausalLM": "hf_model.FogenForCausalLM",
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(output)


if __name__ == "__main__":
    main()
