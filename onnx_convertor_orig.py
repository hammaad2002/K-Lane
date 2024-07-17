import torch
import torch.nn as nn
import onnxruntime
import torch
import numpy as np
import difflib

from myChanges.encoder import Projector # encoder
from myChanges.backbone import VitSegNet # backbone
from myChanges.head import RowSharNotReducRef # head

resnet='resnet34'
pretrained=False
replace_stride_with_dilation=[False, True, False]
out_conv=True
in_channels=[64, 128, 256, -1]
featuremap_out_channel = 64
my_projector = Projector(
    resnet=resnet,
    pretrained=pretrained,
    replace_stride_with_dilation=replace_stride_with_dilation,
    out_conv=out_conv,
    in_channels=in_channels,
    featuremap_out_channel=featuremap_out_channel
)

image_size=144
patch_size=8
channels=64
dim=512
depth=3
heads=16
output_channels=1024
expansion_factor=4
dim_head=64
dropout=0.0
emb_dropout=0.0
is_with_shared_mlp=False
my_vitSegNet = VitSegNet(
    image_size=image_size,
    patch_size=patch_size,
    channels=channels,
    dim=dim,
    depth=depth,
    heads=heads,
    output_channels=output_channels,
    expansion_factor=expansion_factor,
    dim_head=dim_head,
    dropout=dropout,
    emb_dropout=emb_dropout,
    is_with_shared_mlp=is_with_shared_mlp,
)

dim_feat=8 # input feat channels
row_size=144
dim_shared=512
lambda_cls=1.0
thr_ext=0.3
off_grid=2
dim_token=1024
tr_depth=1
tr_heads=16
tr_dim_head=64
tr_mlp_dim=2048
tr_dropout=0.0
tr_emb_dropout=0.0
is_reuse_same_network=False
conf_thr=0.5
cls_lane_color=[(0, 0, 255),
                (0, 50, 255),
                (0, 255, 255),
                (0, 255, 0),
                (255, 0, 0),
                (255, 0, 100)]
my_rowSharNotReducRef = RowSharNotReducRef(
    dim_feat=dim_feat,
    row_size=row_size,
    dim_shared=dim_shared,
    lambda_cls=lambda_cls,
    thr_ext=thr_ext,
    off_grid=off_grid,
    dim_token=dim_token,
    tr_depth=tr_depth,
    tr_heads=tr_heads,
    tr_dim_head=tr_dim_head,
    tr_mlp_dim=tr_mlp_dim,
    tr_dropout=tr_dropout,
    tr_emb_dropout=tr_emb_dropout,
    is_reuse_same_network=is_reuse_same_network,
    conf_thr=conf_thr,
    cls_lane_color=cls_lane_color,
)

from original_model.encoder import Projector # encoder
from original_model.backbone import VitSegNet # backbone
from original_model.head import RowSharNotReducRef # head

resnet='resnet34'
pretrained=False
replace_stride_with_dilation=[False, True, False]
out_conv=True
in_channels=[64, 128, 256, -1]
featuremap_out_channel = 64
orig_projector = Projector(
    resnet=resnet,
    pretrained=pretrained,
    replace_stride_with_dilation=replace_stride_with_dilation,
    out_conv=out_conv,
    in_channels=in_channels,
    featuremap_out_channel=featuremap_out_channel
)

image_size=144
patch_size=8
channels=64
dim=512
depth=3
heads=16
output_channels=1024
expansion_factor=4
dim_head=64
dropout=0.0
emb_dropout=0.0
is_with_shared_mlp=False
orig_vitSegNet = VitSegNet(
    image_size=image_size,
    patch_size=patch_size,
    channels=channels,
    dim=dim,
    depth=depth,
    heads=heads,
    output_channels=output_channels,
    expansion_factor=expansion_factor,
    dim_head=dim_head,
    dropout=dropout,
    emb_dropout=emb_dropout,
    is_with_shared_mlp=is_with_shared_mlp,
)

dim_feat=8 # input feat channels
row_size=144
dim_shared=512
lambda_cls=1.0
thr_ext=0.3
off_grid=2
dim_token=1024
tr_depth=1
tr_heads=16
tr_dim_head=64
tr_mlp_dim=2048
tr_dropout=0.0
tr_emb_dropout=0.0
is_reuse_same_network=False
conf_thr=0.5
cls_lane_color=[(0, 0, 255),
                (0, 50, 255),
                (0, 255, 255),
                (0, 255, 0),
                (255, 0, 0),
                (255, 0, 100)]
orig_rowSharNotReducRef = RowSharNotReducRef(
    dim_feat=dim_feat,
    row_size=row_size,
    dim_shared=dim_shared,
    lambda_cls=lambda_cls,
    thr_ext=thr_ext,
    off_grid=off_grid,
    dim_token=dim_token,
    tr_depth=tr_depth,
    tr_heads=tr_heads,
    tr_dim_head=tr_dim_head,
    tr_mlp_dim=tr_mlp_dim,
    tr_dropout=tr_dropout,
    tr_emb_dropout=tr_emb_dropout,
    is_reuse_same_network=is_reuse_same_network,
    conf_thr=conf_thr,
    cls_lane_color=cls_lane_color,
)

class K_lane_changed(nn.Module):
    def __init__(self):
        super(K_lane_changed, self).__init__()
        self.encoder = my_projector
        self.backbone = my_vitSegNet
        self.head = my_rowSharNotReducRef
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        print(x.shape)
        x1, x2, x3, x4 = self.head(x)
        return x1, x2, x3, x4
    
class K_lane_orig(nn.Module):
    def __init__(self):
        super(K_lane_orig, self).__init__()
        self.encoder = orig_projector
        self.backbone = orig_vitSegNet
        self.head = orig_rowSharNotReducRef
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        print(x.shape)
        x1, x2, x3, x4 = self.head(x)
        return x1, x2, x3, x4

input_temp = torch.rand(1, 3, 1152, 1152)

# FUCKING PREPROCESSING THE STUPID KEYS (I HATE REGISTRATION METHOD THEY USED)
initial_weights = torch.load('Proj28_GFC-T3_RowRef_82_73.pth', map_location=torch.device('cpu'))['net']

for k in list(initial_weights.keys()):
    if k == "module.conv1.weight" or k == "module.conv1.bias":
        initial_weights.pop(k)
    elif k.startswith('module.pcencoder'):
        new_k = k.replace('module.pcencoder', 'encoder')
        initial_weights[new_k] = initial_weights.pop(k)
    elif k.startswith('module.backbone'):
        new_k = k.replace('module.backbone', 'backbone')
        initial_weights[new_k] = initial_weights.pop(k)
    elif k.startswith('module.heads'):
        new_k = k.replace('module.heads', 'head')
        initial_weights[new_k] = initial_weights.pop(k)

model_changed = K_lane_changed()

# Get the state dictionary
state_dict = model_changed.state_dict()

# 2nd stage processing
model_keys = list(state_dict.keys())
initial_keys = list(initial_weights.keys())

for k in initial_keys:
    # Find the best match in model_keys
    best_match = difflib.get_close_matches(k, model_keys, n=1, cutoff=0.6)
    if best_match:
        # Get the single best match
        new_k = best_match[0]

        # Replace 'module.heads' with 'head' in the new key
        new_k = new_k.replace(k, new_k)

        model_keys.remove(new_k)
        temp = initial_weights.pop(k)

        # Update initial_weights with the new key
        initial_weights[new_k] = temp

model_changed.load_state_dict(initial_weights, strict=False)
model_changed.eval()  # Set the model to evaluation mode

model_orig = K_lane_orig()
model_orig.eval()  # Set the model to evaluation mode

# Define dynamic axes
dynamic_axes = {'input': {0: 'batch_size'},
                'output_0': {0: 'batch_size'},
                'output_1': {0: 'batch_size'},
                'output_2': {0: 'batch_size'},
                'output_3': {0: 'batch_size'}}

# Define names for input and output of model
input_names = ["input"]
output_names = ["output_0", "output_1", "output_2", "output_3"]

# Convert the pytorch to its script version
scripted_model_changed = torch.jit.script(model_changed)

# Model name
onnx_model_path_changed = 'K_Lane_changed.onnx'
onnx_model_path_orig = 'K_Lane_orig.onnx'

# Export the model using script model
torch.onnx.export(scripted_model_changed,
                    input_temp, 
                    onnx_model_path_changed,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    do_constant_folding=True,
                    export_params=True,
                    verbose=False)

print('My changed version of model converted to ONNX successfully!')

# scripted_model_orig = torch.jit.script(model_orig)
# torch.onnx.export(scripted_model_orig,
#                     input_temp, 
#                     onnx_model_path_orig,
#                     input_names=input_names,
#                     output_names=output_names,
#                     dynamic_axes=dynamic_axes,
#                     opset_version=17,
#                     do_constant_folding=True,
#                     export_params=True,
#                     verbose=False)

# print('My original version of model converted to ONNX successfully!')