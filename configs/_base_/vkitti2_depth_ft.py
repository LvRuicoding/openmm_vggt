# Common mmengine-style config for VKITTI2 depth fine-tuning.

data_root = '/home/dataset-local/lr/code/vggt/data/vkitti'
image_size = (280, 518)
num_frames = 12
frame_stride = 1

train_dataset = dict(
    type='VKITTI2SequenceDataset',
    root=data_root,
    split='train',
    num_frames=num_frames,
    stride=frame_stride,
    image_size=image_size,
    variations=['clone'],
    cameras=['Camera_0'],
)

val_dataset = dict(
    type='VKITTI2SequenceDataset',
    root=data_root,
    split='val',
    num_frames=num_frames,
    stride=frame_stride,
    image_size=image_size,
    variations=['clone'],
    cameras=['Camera_0'],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=train_dataset,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=val_dataset,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

optimizer = dict(
    type='AdamW',
    lr=5e-6,
    weight_decay=0.05,
)

scheduler = dict(
    type='CosineAnnealingLR',
    T_max=24,
    eta_min=5e-7,
)

checkpoint = ''
output_dir = '/home/dataset-local/lr/code/openmm_vggt/trainoutput/vkitti2_depth_ft'
epochs = 24
depth_weight = 1.0
grad_clip = 1.0
amp = True
save_every = 4
log_interval = 10
seed = 42
freeze_backbone = False
