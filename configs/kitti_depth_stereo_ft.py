# mmengine-style Python config for KITTI depth completion fine-tuning
# with stereo + temporal context (3 time steps × 2 cameras = 6 images).
#
# Input layout per sample (S=6, cam_num=6, f_num=1):
#   [t0_cam02, t0_cam03, t1_cam02, t1_cam03, t2_cam02, t2_cam03]
#
# Only the last time step image_02 (t2_cam02) carries GT depth supervision.
# When history is insufficient (e.g. first frame of a sequence), earlier
# slots are filled by repeating the earliest available frame (padding).
# All other prediction heads remain active in the model.
#
# Image resolution: (280, 518) — matches demo_viser_nusc_eval_norm.py
# preprocessing: H=280=20×14, W=518=37×14.
#
# Run (single GPU):
#   python tools/train.py configs/kitti_depth_stereo_ft.py --no-eval
# Run (4 GPUs):
#   torchrun --nproc_per_node=4 tools/train.py configs/kitti_depth_stereo_ft.py --no-eval

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
depth_root = '/home/dataset-local/lr/code/openmm_vggt/data/OpenDataLab___KITTI_depth_completion'
raw_root   = '/home/dataset-local/lr/code/openmm_vggt/data/kitti_raw'

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# cam_num = n_time_steps * 2 = 3 * 2 = 6  →  f_num = S // cam_num = 6 // 6 = 1
model = dict(
    type='VGGT_decoder_global',
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    enable_camera=True,
    enable_point=False,
    enable_depth=True,
    enable_track=False,
    cam_num=6,
)

# ---------------------------------------------------------------------------
# Image resolution — matches the preprocessing used in demo_viser_nusc_eval_norm.py
# (load_and_preprocess_images_pos hardcodes H=280, W=518)
# Both are multiples of 14: 280 = 20×14, 518 = 37×14
# ---------------------------------------------------------------------------
image_size = (280, 518)    # (H, W) — closest multiples of 14 to native KITTI resolution (375, 1242): 27×14=378, 89×14=1246
n_time_steps = 3           
stride = 1

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

# ---------------------------------------------------------------------------
# Validation dataset  (same folder structure, split='val')
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
scheduler = dict(
    type='CosineAnnealingLR',
    T_max=30,
    eta_min=2e-6,
)

# ---------------------------------------------------------------------------
# Loss weights
# Depth is the only supervised signal; camera/point/track weights are 0
# so their losses are not back-propagated, but the heads stay active.
# ---------------------------------------------------------------------------
depth_weight           = 1.0
camera_weight          = 0.0
pose_translation_weight = 0.0
pose_rotation_weight   = 0.0
pose_fov_weight        = 0.0

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
checkpoint = '/home/dataset-local/lr/code/openmm_vggt/ckpt/checkpoint_5.pt'
output_dir = '/home/dataset-local/lr/code/openmm_vggt/trainoutput/kitti_depth_stereo_ft'

epochs       = 12
grad_clip    = 1.0
amp          = True
save_every   = 2      # save checkpoint every 3 epochs
log_interval = 10
seed         = 42

# Keep backbone and all heads except depth_head frozen throughout training.
# Only depth_head (~10M params) is trained, which is much faster.
freeze_modules            = ('aggregator', 'camera_head', 'camera_relative_head', 'mv_blocks', 'rel_pose_embed', 'batch_norm', 'layer_norm')
freeze_modules_for_epochs = 30   # = epochs, freeze for the entire training run
