"""
Models Module
=============
Model architectures for Sentinel Stroke pipeline.

Note: Imports are done on-demand to avoid dependency errors
when MONAI is not installed locally (training runs on Lambda Labs).
"""

# Lazy imports - import when needed
def create_swin_unetr(*args, **kwargs):
    from .swin_unetr import create_swin_unetr as _create
    return _create(*args, **kwargs)

def create_segresnet(*args, **kwargs):
    from .segresnet import create_segresnet as _create
    return _create(*args, **kwargs)

def create_fusion_network(*args, **kwargs):
    from .fusion_network import create_fusion_network as _create
    return _create(*args, **kwargs)

__all__ = [
    'create_swin_unetr',
    'create_segresnet',
    'create_fusion_network',
]

