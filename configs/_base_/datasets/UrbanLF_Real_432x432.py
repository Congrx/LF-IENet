# dataset settings
dataset_type = 'UrbanLFRealDataset'
data_root = './data/UrbanLF/UrbanLF_Real/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (432, 432)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(623, 432), ratio_range=(0.5, 2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'sequence_imgs', 'gt_semantic_seg','disparity','sequence_index']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(623, 432),
        img_ratios=[1.0],
        flip=False,
        #img_ratios=[1.0,0.75,1.5,1.25],
        #flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'sequence_imgs','disparity','sequence_index']),
            dict(type='Collect', keys=['img', 'sequence_imgs','disparity','sequence_index']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        img_dir=data_root,
        ann_dir=data_root,
        sai_dir=data_root,
        dis_dir=data_root,
        sai_number=4,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        img_dir=data_root,
        ann_dir=data_root,
        sai_dir=data_root,
        dis_dir=data_root,
        sai_number=4,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        img_dir=data_root,
        ann_dir=data_root,
        sai_dir=data_root,
        dis_dir=data_root,
        sai_number=4,
        pipeline=test_pipeline))
