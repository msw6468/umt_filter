import clip.clip as clip
import os
import torch
from collections import OrderedDict

path = '/gallery_moma/sangwoo.moon/umt_filter/multi_modality/pretrained_model/clip_visual_encoder' # SNU server
# path = '/net/nfs3.prior/dongjook/pretrained_model/clip_visual_encoder' # AI2 server
os.makedirs(path, exist_ok=True)

model, _ = clip.load("ViT-L/14", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(path, 'vit_l14.pth'))
