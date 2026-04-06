# mmengine-style Python config for KITTI depth completion fine-tuning.
# This is the retained KITTI depth stereo baseline config.
# pred_depth is scaled by depth_pred_scale before loss computation.
# depth_pred_scale=20 compensates for the original model's output range
# (~0.8m mean) vs KITTI's typical depth range (~15-20m mean).
#
# Run (4 GPUs):
#   torchrun --nproc_per_node=4 tools/train.py configs/kitti_depth_stereo_ft_scaled.py --no-eval

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
depth_root = '/home/dataset-local/lr/code/openmm_vggt/data/OpenDataLab___KITTI_depth_completion'
raw_root   = '/home/dataset-local/lr/code/openmm_vggt/data/kitti_raw'

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = dict(
    type='VGGT_decoder_global',
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    enable_camera=True,
    enable_point=False,
    enable_depth=True,
    enable_track=False,
    cam_num=2,  # Stereo pair: left (image_02) and right (image_03)
)

image_size   = (280, 518)
n_time_steps = 3
stride       = 1

# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------
train_dataset = dict(
    type='KITTIDepthCompletionStereoDataset',
    depth_root=depth_root,
    raw_root=raw_root,
    split='train',
    n_time_steps=n_time_steps,
    stride=stride,
    image_size=image_size,
    strict=False,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
)

val_dataset = dict(
    type='KITTIDepthCompletionStereoDataset',
    depth_root=depth_root,
    raw_root=raw_root,
    split='val',
    n_time_steps=n_time_steps,
    stride=stride,
    image_size=image_size,
    strict=False,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
)

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
optimizer = dict(
    type='AdamW',
    lr=2e-5,
    weight_decay=0.05,
)

scheduler = dict(
    type='CosineAnnealingLR',
    T_max=30,
    eta_min=2e-6,
)

# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------
depth_weight            = 1.0
camera_weight           = 0.0
pose_translation_weight = 0.0
pose_rotation_weight    = 0.0
pose_fov_weight         = 0.0

# Scale factor applied to model depth output before loss computation.
# pred_depth_for_loss = pred_depth * depth_pred_scale
# This compensates for the original model outputting ~0.8m mean
# while KITTI GT has ~15-20m mean.
depth_pred_scale = 20.0

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
checkpoint = '/home/dataset-local/lr/code/openmm_vggt/ckpt/checkpoint_5.pt'
output_dir = '/home/dataset-local/lr/code/openmm_vggt/trainoutput/kitti_depth_stereo_ft_scaled'

epochs       = 4
grad_clip    = 1.0
amp          = True
save_every   = 2
log_interval = 10
seed         = 42

freeze_modules            = ('aggregator', 'camera_head', 'camera_relative_head', 'mv_blocks', 'rel_pose_embed', 'batch_norm', 'layer_norm')
freeze_modules_for_epochs = 4
