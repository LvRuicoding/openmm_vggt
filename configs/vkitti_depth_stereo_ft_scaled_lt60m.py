# mmengine-style Python config for vKITTI stereo depth fine-tuning (<60m only).
#
# This config is based on configs/vkitti_depth_stereo_ft_scaled.py and is meant
# for the setup where both training loss and evaluation metrics only consider
# pixels with GT depth < 60m.
#
# Current codebase notes:
# - Evaluation is already supported via tools/eval_vkitti_depth_lt60m.py.
# - Training code does not yet consume train_max_gt_depth_m automatically; this
#   config stores the intended threshold so the later training-side change can
#   read it directly.
#
# Run (4 GPUs):
#   torchrun --nproc_per_node=4 tools/train.py configs/vkitti_depth_stereo_ft_scaled_lt60m.py
#
# Eval:
#   torchrun --nproc_per_node=4 tools/eval_vkitti_depth_lt60m.py configs/vkitti_depth_stereo_ft_scaled_lt60m.py

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
vkitti_root = '/home/dataset-local/lr/code/openmm_vggt/data/vkitti'

train_scenes = ('Scene01', 'Scene02', 'Scene06', 'Scene18')
val_scenes   = ('Scene20',)

# ---------------------------------------------------------------------------
# Depth range policy
# ---------------------------------------------------------------------------
max_gt_depth_m = 60.0
train_max_gt_depth_m = max_gt_depth_m
eval_max_gt_depth_m = max_gt_depth_m

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
    cam_num=2,
)

image_size   = (280, 518)
n_time_steps = 3
stride       = 1

# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------
train_dataset = dict(
    type='VKITTIDepthStereoDataset',
    root=vkitti_root,
    split='train',
    scene_names=train_scenes,
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
    type='VKITTIDepthStereoDataset',
    root=vkitti_root,
    split='val',
    scene_names=val_scenes,
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

depth_pred_scale = 20.0

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
checkpoint = '/home/dataset-local/lr/code/openmm_vggt/ckpt/checkpoint_5.pt'
output_dir = '/home/dataset-local/lr/code/openmm_vggt/trainoutput/vkitti_depth_stereo_ft_scaled_lt60m'

epochs       = 24
grad_clip    = 1.0
amp          = True
save_every   = 2
log_interval = 10
seed         = 42

freeze_modules            = ('aggregator', 'camera_head', 'camera_relative_head', 'mv_blocks', 'rel_pose_embed', 'batch_norm', 'layer_norm')
freeze_modules_for_epochs = 24
