"""
Loads the braindecode-bendr pretrained checkpoint with CHB-MIT's channel
set, via InterpolatedBENDR. Works around two version-skew bugs in the
HuggingFace config.json relative to the installed braindecode version:
1. Obsolete keys (n_chans_pretrained, chan_proj_max_norm) no longer
   accepted by BENDR's constructor.
2. 'activation' stored as a string path instead of an actual class.
"""
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from braindecode.models import InterpolatedBENDR


def load_pretrained_interpolated_bendr(chs_info, n_outputs=2):
    config_path = hf_hub_download(repo_id="braindecode/braindecode-bendr", filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    config.pop("n_chans_pretrained", None)
    config.pop("chan_proj_max_norm", None)
    config.pop("n_times", None)
    config["n_outputs"] = n_outputs
    config["n_chans"] = len(chs_info)
    config["chs_info"] = chs_info

    if isinstance(config.get("activation"), str):
        config["activation"] = torch.nn.GELU

    model = InterpolatedBENDR.from_config(config)

    weights_path = hf_hub_download(repo_id="braindecode/braindecode-bendr", filename="model.safetensors")
    state_dict = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        print(f"WARNING: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    else:
        print(f"InterpolatedBENDR loaded cleanly: {sum(p.numel() for p in model.parameters())} params")

    return model

# """
# Loads the braindecode-bendr pretrained checkpoint, working around a
# version-skew bug where the HF config.json still contains keys
# (n_chans_pretrained, chan_proj_max_norm) removed from braindecode 1.8.1's
# BENDR class.
# """
# import json
# from huggingface_hub import hf_hub_download
# from safetensors.torch import load_file
# from braindecode.models import BENDR


# def load_pretrained_bendr(n_outputs=2, chs_info=None):
#     config_path = hf_hub_download(repo_id="braindecode/braindecode-bendr", filename="config.json")
#     with open(config_path) as f:
#         config = json.load(f)

#     config.pop("n_chans_pretrained", None)
#     config.pop("chan_proj_max_norm", None)
#     config["n_outputs"] = n_outputs
#     if chs_info is not None:
#         config["chs_info"] = chs_info

#     model = BENDR.from_config(config)

#     weights_path = hf_hub_download(repo_id="braindecode/braindecode-bendr", filename="model.safetensors")
#     state_dict = load_file(weights_path)
#     missing, unexpected = model.load_state_dict(state_dict, strict=False)

#     if missing or unexpected:
#         print(f"WARNING: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
#     else:
#         print(f"BENDR checkpoint loaded cleanly: {sum(p.numel() for p in model.parameters())} params")

#     return model