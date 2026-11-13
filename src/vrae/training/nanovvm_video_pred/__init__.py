"""First-frame conditioned 16-frame future prediction on NanoVVM clip manifests."""

from vrae.training.nanovvm_video_pred.data import (
    FRAME_INTERVAL,
    FUTURE_FRAMES,
    RGB_SEQ_LEN,
    NanoVVMVideoPredictionDataset,
    split_condition_future,
)
from vrae.training.nanovvm_video_pred.manifest import ClipManifest, load_clip_manifest

__all__ = [
    "ClipManifest",
    "FRAME_INTERVAL",
    "FUTURE_FRAMES",
    "NanoVVMVideoPredictionDataset",
    "RGB_SEQ_LEN",
    "load_clip_manifest",
    "split_condition_future",
]
