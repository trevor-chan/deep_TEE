"""
MViT : Multiscale Vision Transformer. Based on 'MViTv2: Improved Multiscale Vision Transformers for Classification and Detection' and 'Multiscale Vision Transformers'
"""
import torch
import torchvision
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

@dataclass
class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: List[int]
    kernel_kv: List[int]
    stride_q: List[int]
    stride_kv: List[int]

def fetch_vit_model(
    spatial_size = (150,200),
    temporal_size = 32,
    residual_pool = False,
    residual_with_cls_embed = False,
    stochastic_depth_prob = 0.0,
    rel_pos_embed = True,
    proj_after_attn = True,
    num_classes = 1,
):


    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "output_channels": [192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768, 768],
        "kernel_q": [[], [3, 3, 3], [], [3, 3, 3], [], [], [], [], [], [], [], [], [], [], [3, 3, 3], []],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [[], [1, 2, 2], [], [1, 2, 2], [], [], [], [], [], [], [], [], [], [], [1, 2, 2], []],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    model = torchvision.models.video.MViT(
            spatial_size = spatial_size,
            temporal_size = temporal_size,
            block_setting = block_setting,
            residual_pool = residual_pool,
            residual_with_cls_embed = residual_with_cls_embed,
            stochastic_depth_prob = stochastic_depth_prob,
            rel_pos_embed = rel_pos_embed,
            proj_after_attn = proj_after_attn,
            num_classes = 1,
        )
    
    return model






def fetch_s3d_model(
    num_classes: int = 1,
    dropout: float = 0.2,
):
    return torchvision.models.video.s3d(num_classes=num_classes, dropout=dropout)


def r3d_18_noweights(num_classes:int = 1) -> torchvision.models.video.VideoResNet:
    """Construct 18 layer Resnet3D model."""

    return torchvision.models.video.VideoResNet(
        torchvision.models.video.resnet.BasicBlock,
        [torchvision.models.video.resnet.Conv3DSimple] * 4,
        [2, 2, 2, 2],
        torchvision.models.video.resnet.BasicStem,
        num_classes = num_classes,
    )

def fetch_resnet_model(
    num_classes: int = 1,
    dropout: float = 0.2,
):       
    return r3d_18_noweights(num_classes=num_classes)