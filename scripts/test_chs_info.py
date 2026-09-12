from braindecode.models import InterpolatedBENDR
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import torch
import sys
sys.path.insert(0, 'scripts')
from chbmit_chs_info import get_chbmit_chs_info

chs_info = get_chbmit_chs_info()

config_path = hf_hub_download(repo_id='braindecode/braindecode-bendr', filename='config.json')
with open(config_path) as f:
    config = json.load(f)

config.pop('n_chans_pretrained', None)
config.pop('chan_proj_max_norm', None)
config['n_outputs'] = 2
config['n_chans'] = len(chs_info)
config['chs_info'] = chs_info
config.pop('n_times', None)

# Fix: activation is stored as a string path, but the model expects
# the actual class object
if isinstance(config.get('activation'), str):
    config['activation'] = torch.nn.GELU

model = InterpolatedBENDR.from_config(config)
print('InterpolatedBENDR constructed successfully')

weights_path = hf_hub_download(repo_id='braindecode/braindecode-bendr', filename='model.safetensors')
state_dict = load_file(weights_path)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f'Missing keys: {len(missing)}')
print(f'Unexpected keys: {len(unexpected)}')
if missing:
    print('Missing (first 10):', missing[:10])
print(f'Total params: {sum(p.numel() for p in model.parameters())}')
