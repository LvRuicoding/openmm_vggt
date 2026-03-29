# mmengine-style Python config for KITTI depth fine-tuning

# --------------- model ---------------
model = dict(
    type='VGGT_decoder_global',
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    enable_camera=False,
    enable_point=False,
    enable_depth=True,
    enable_track=False,
    cam_num=1,
)

# --------------- data ---------------
depth_root = '/home/dataset-local/lr/code/openmm_vggt/data/OpenDataLab___KITTI_depth_completion'
raw_root = '/home/dataset-local/lr/code/openmm_vggt/data/kitti_raw'
image_size = (280, 518)
num_frames = 6
frame_stride = 1

train_dataset = dict(
    type='KITTIDepthCompletionSequenceDataset',
    depth_root=depth_root,
    raw_root=raw_root,
    split='train',
    num_frames=num_frames,
    stride=frame_stride,
    image_size=image_size,
    cameras=('image_02', 'image_03'),
    strict=True,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=train_dataset,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
)

val_dataset = None
val_dataloader = None

# --------------- optimiser ---------------
optimizer = dict(
    type='AdamW',
    lr=5e-6,
    weight_decay=0.05,
)

# --------------- scheduler ---------------
scheduler = dict(
    type='CosineAnnealingLR',
    T_max=12,
    eta_min=5e-7,
)

# --------------- training ---------------
checkpoint = '/home/dataset-local/lr/code/openmm_vggt/ckpt/checkpoint_5.pt'
output_dir = '/home/dataset-local/lr/code/openmm_vggt/trainoutput/kitti_depth_only_ft'
epochs = 12
depth_weight = 1.0
camera_weight = 0.0
pose_translation_weight = 0.0
pose_rotation_weight = 0.0
pose_fov_weight = 0.0
grad_clip = 1.0
amp = True
save_every = 4
log_interval = 10
seed = 42
freeze_backbone = False
