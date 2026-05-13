# Left-camera FLoSP variant of the occupancy-only KITTI Semantic config.

_base_ = "./kitti_semantic_occ_mix_window_attn_early_ft_monoscene_head_cp_364x1218_occ_only.py"

model = dict(
    # Dataset order is ("image_02", "image_03") within each time step, so 0 selects image_02.
    occupancy_view_indices=(0,),
)

train_dataset = dict(
    occupancy_cache_dir="/tmp/openmm_vggt_kitti_semantic_occ_only_left_flosp_cache_train",
)

val_dataset = dict(
    occupancy_cache_dir="/tmp/openmm_vggt_kitti_semantic_occ_only_left_flosp_cache_val",
)

output_dir = "/home/dataset-local/lr/code/openmm_vggt/trainoutput/kitti_semantic_occ_mix_window_attn_early_ft_monoscene_head_cp_364x1218_occ_only_left_flosp"
