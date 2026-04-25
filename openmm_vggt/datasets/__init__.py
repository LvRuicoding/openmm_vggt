from .kitti_depth_stereo import KITTIDepthCompletionStereoDataset
from .kitti_semantic_occ import KITTISemanticOccupancyDataset
from .vkitti_depth_stereo import VKITTIDepthStereoDataset

try:
    from .ddad_depth_temporal import DDADDepthTemporalDataset
    from .ddad_depth_temporal_masked_max_pool import DDADDepthTemporalMaskedMaxPoolDataset
except Exception:  # optional dependency path for DGP/DDAD
    DDADDepthTemporalDataset = None
    DDADDepthTemporalMaskedMaxPoolDataset = None

__all__ = [
    "KITTIDepthCompletionStereoDataset",
    "KITTISemanticOccupancyDataset",
    "VKITTIDepthStereoDataset",
]

if DDADDepthTemporalDataset is not None:
    __all__.append("DDADDepthTemporalDataset")
if DDADDepthTemporalMaskedMaxPoolDataset is not None:
    __all__.append("DDADDepthTemporalMaskedMaxPoolDataset")
