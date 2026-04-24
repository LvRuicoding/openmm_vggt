# mmengine-style Python config for DDAD 3-camera depth fine-tuning.
# This variant keeps the DDAD temporal setup but restricts training to
# CAMERA_01 / CAMERA_05 / CAMERA_06 to mimic a reduced multi-camera setup.

ddad_scene_dataset_json = "/home/dataset-local/lr/code/openmm_vggt/data/ddad/ddad_train_val/ddad.json"

camera_names = ("CAMERA_01", "CAMERA_05", "CAMERA_06")

model = dict(
    type="mix_decoder_global_window_attn_early",
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    enable_camera=True,
    enable_point=False,
    enable_depth=True,
    enable_track=False,
    cam_num=3,
    voxel_size=(0.4, 0.4, 0.2),
    point_cloud_range=(-100.0, -100.0, -5.0, 100.0, 100.0, 3.0),
    serializer_grid_size_2d=14.0,
    use_top_k=False,
    top_k_per_patch=5,
    fusion_window_size=(6, 6),
    fusion_shift_size=(3, 3),
    fusion_num_heads=16,
    fusion_mlp_ratio=4.0,
    fusion_attn_backend="auto",
)

image_size = (350, 560)
n_time_steps = 3
stride = 1

train_dataset = dict(
    type="DDADDepthTemporalDataset",
    root=ddad_scene_dataset_json,
    split="train",
    camera_names=camera_names,
    n_time_steps=n_time_steps,
    stride=stride,
    image_size=image_size,
    strict=False,
    return_lidar=True,
    max_lidar_points=32768,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=32,
    pin_memory=True,
)

val_dataset = dict(
    type="DDADDepthTemporalDataset",
    root=ddad_scene_dataset_json,
    split="val",
    camera_names=camera_names,
    n_time_steps=n_time_steps,
    stride=stride,
    image_size=image_size,
    strict=False,
    return_lidar=True,
    max_lidar_points=32768,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
)

optimizer = dict(
    type="AdamW",
    lr=2e-5,
    weight_decay=0.05,
)

scheduler = dict(
    type="CosineAnnealingLR",
    T_max=30,
    eta_min=2e-6,
)

depth_weight = 1.0
camera_weight = 0.0
pose_translation_weight = 0.0
pose_rotation_weight = 0.0
pose_fov_weight = 0.0
depth_supervision_source = "batch_depth"

depth_pred_scale = 20.0

checkpoint = "/home/dataset-local/lr/code/openmm_vggt/ckpt/checkpoint_5.pt"
output_dir = "/home/dataset-local/lr/code/openmm_vggt/trainoutput/ddad_depth_3cam_010506_mix_window_attn_early_ft"

epochs = 4
grad_clip = 1.0
amp = True
save_every = 1
log_interval = 10
seed = 42

freeze_modules = (
    "aggregator",
    "camera_head",
    "camera_relative_head",
    "mv_blocks",
    "rel_pose_embed",
    "batch_norm",
    "layer_norm",
)
freeze_modules_for_epochs = 4
