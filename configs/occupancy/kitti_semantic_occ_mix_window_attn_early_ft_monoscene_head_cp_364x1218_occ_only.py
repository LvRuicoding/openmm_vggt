# mmengine-style config for KITTI semantic voxel occupancy-only fine-tuning

semantic_root = "/home/dataset-local/lr/code/openmm_vggt/data/kitti_semantic"
raw_root = "/home/dataset-local/lr/code/openmm_vggt/data/kitti_raw"
dense_voxel_root = "/home/dataset-local/lr/code/openmm_vggt/data/odemetry_voxels"

image_size = (364, 1218)
n_time_steps = 3
stride = 1

fusion_voxel_size = (0.4, 0.4, 0.8)
occ_voxel_size = (0.2, 0.2, 0.2)
occ_point_cloud_range = (0.0, -25.6, -2.0, 51.2, 25.6, 4.4)

model = dict(
    type="mix_decoder_global_window_attn_early_occ",
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    enable_camera=False,
    enable_point=False,
    enable_depth=False,
    enable_track=False,
    cam_num=2,
    voxel_size=fusion_voxel_size,
    point_cloud_range=occ_point_cloud_range,
    serializer_grid_size_2d=14.0,
    use_top_k=False,
    fusion_window_size=(10, 10),
    fusion_shift_size=(5, 5),
    fusion_num_heads=16,
    fusion_mlp_ratio=4.0,
    fusion_attn_backend="auto",
    occupancy_head=dict(
        type="MonoSceneOccupancyHead",
        voxel_size=occ_voxel_size,
        point_cloud_range=occ_point_cloud_range,
        num_classes=20,
        feature=64,
        project_scale=2,
        context_prior=True,
        n_relations=4,
    ),
)

train_dataset = dict(
    type="KITTISemanticOccupancyDataset",
    semantic_root=semantic_root,
    raw_root=raw_root,
    split="train",
    n_time_steps=n_time_steps,
    stride=stride,
    image_size=image_size,
    strict=False,
    max_lidar_points=32768,
    voxel_size=occ_voxel_size,
    point_cloud_range=occ_point_cloud_range,
    dense_voxel_root=dense_voxel_root,
    require_dense_voxel_target=True,
    occupancy_cache_dir="/tmp/openmm_vggt_kitti_semantic_occ_only_cache_train",
    frustum_size=4,
    color_jitter=(0.4, 0.4, 0.4),
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
)

val_dataset = dict(
    type="KITTISemanticOccupancyDataset",
    semantic_root=semantic_root,
    raw_root=raw_root,
    split="val",
    n_time_steps=n_time_steps,
    stride=stride,
    image_size=image_size,
    strict=False,
    max_lidar_points=32768,
    voxel_size=occ_voxel_size,
    point_cloud_range=occ_point_cloud_range,
    dense_voxel_root=dense_voxel_root,
    require_dense_voxel_target=True,
    occupancy_cache_dir="/tmp/openmm_vggt_kitti_semantic_occ_only_cache_val",
    frustum_size=4,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    pin_memory=True,
)

optimizer = dict(
    type="AdamW",
    lr=2e-5,
    weight_decay=0.05,
)
random_init_lr_multiplier = 5.0

scheduler = dict(
    type="CosineAnnealingLR",
    T_max=24,
    eta_min=2e-6,
)

depth_weight = 0.0
camera_weight = 0.0
pose_translation_weight = 0.0
pose_rotation_weight = 0.0
pose_fov_weight = 0.0
depth_pred_scale = 20.0

occupancy_weight = 1.0
occupancy_num_classes = 20
occupancy_ignore_index = 255
occupancy_class_weights = [
    0.0446,  # empty
    0.0603,  # car
    0.0852,  # bicycle
    0.0856,  # motorcycle
    0.0747,  # truck
    0.0734,  # other-vehicle
    0.0801,  # person
    0.0796,  # bicyclist
    0.0818,  # motorcyclist
    0.0557,  # road
    0.0653,  # parking
    0.0568,  # sidewalk
    0.0683,  # other-ground
    0.0560,  # building
    0.0603,  # fence
    0.0530,  # vegetation
    0.0688,  # trunk
    0.0574,  # terrain
    0.0716,  # pole
    0.0786,  # traffic-sign
]
sem_scal_weight = 1.0
geo_scal_weight = 1.0
frustum_proportion_weight = 1.0
context_prior_weight = 1.0

checkpoint = "/home/dataset-local/lr/code/openmm_vggt/ckpt/checkpoint_5.pt"
output_dir = "/home/dataset-local/lr/code/openmm_vggt/trainoutput/kitti_semantic_occ_mix_window_attn_early_ft_monoscene_head_cp_364x1218_occ_only"

epochs = 8
grad_clip = 1.0
amp = True
save_every = 2
log_interval = 10
seed = 42

freeze_modules = (
    "aggregator",
    "rel_pose_embed",
    "layer_norm",
    "batch_norm",
    "mv_blocks",
)
freeze_modules_for_epochs = 2
