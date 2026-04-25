# KITTI Semantic Occupancy 增量模块说明

这份笔记整理的是：

- `configs/occupancy/kitti_semantic_occ_mix_window_attn_early_ft.py`
- 相对 `configs/early/kitti_depth_stereo_mix_window_attn_early_ft.py`
- 在模型设计上新增了哪些模块
- 这些模块的具体实现文件在哪里

## 一句话结论

这个 occupancy 版本不是重写整套模型，而是在原来的 `mix_decoder_global_window_attn_early` 基础上，新增了一个 occupancy 分支，因此它的设计思路是“保留原始 early window attention 主干 + 增量添加 occupancy head 和对应训练逻辑”。

## 1. 配置入口

先看配置文件：

- [configs/occupancy/kitti_semantic_occ_mix_window_attn_early_ft.py](/home/dataset-local/lr/code/openmm_vggt/configs/occupancy/kitti_semantic_occ_mix_window_attn_early_ft.py:13)

这里把模型类型指定成：

```python
type="mix_decoder_global_window_attn_early_occ"
```

并且额外传入了：

```python
occupancy_head=dict(
    voxel_size=occ_voxel_size,
    point_cloud_range=occ_point_cloud_range,
    num_classes=20,
    hidden_dim=16,
    depth_scale=20.0,
)
```

这说明 occupancy 版本的增量入口，首先就是模型类型从原来的：

```python
mix_decoder_global_window_attn_early
```

改成了：

```python
mix_decoder_global_window_attn_early_occ
```

## 2. 增量模型本体

最关键的实现文件在这里：

- [openmm_vggt/models/fusion_layer/mix_decoder_global_window_attn_early_occ.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/models/fusion_layer/mix_decoder_global_window_attn_early_occ.py:15)

这是你要优先看的文件，因为它就是“在原模型上做增量扩展”的主体。

它的核心特征是：

- 直接继承原始模型 `mix_decoder_global_window_attn_early`
- 在 `__init__` 里新增 `OccupancyHead`
- 在 forward 末尾增加 `occupancy_logits` 的生成逻辑

关键代码结构是：

```python
class mix_decoder_global_window_attn_early_occ(mix_decoder_global_window_attn_early):
```

这句基本就说明了设计方式：不是平行重写，而是继承式增量开发。

### 这里新增了什么

#### 2.1 新增 occupancy head 的构造

在这个文件里可以看到：

```python
self.occupancy_head = None if occupancy_head is None else OccupancyHead(
    token_dim=2 * embed_dim,
    patch_size=patch_size,
    **occupancy_head,
)
```

也就是说，原始 early 模型没有这个头，occupancy 版本才额外挂上了它。

#### 2.2 新增 occupancy forward 分支

在 forward 逻辑里，深度输出之后又接了一步：

```python
predictions["occupancy_logits"] = self.occupancy_head(...)
```

它使用的输入包括：

- 最后时刻的 token
- 最后时刻的 depth
- intrinsics
- camera_to_world
- lidar_to_world

所以 occupancy 不是单独从图像直接分类出来，而是借助深度分支提供的几何信息，再投影到 voxel 空间做 3D occupancy 分类。

## 3. 被继承的原始 early 模型

原模型文件在这里：

- [openmm_vggt/models/fusion_layer/mix_decoder_global_window_attn_early.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/models/fusion_layer/mix_decoder_global_window_attn_early.py:13)

这是 occupancy 版本所继承的基类实现。

这个文件主要负责的是原始 backbone 级别的 early fusion 逻辑，比如：

- `EarlyFusionAggregator`
- `ShiftWindowPatchVoxelCrossFusion`
- 早期 patch 和 voxel 的 cross attention
- shifted-window 的融合策略

也就是说，如果你想知道“occupancy 版本保留了什么旧设计”，这个文件就是最直接的参考。

## 4. 更底层的公共基类

如果你还想继续往下追原模型的共用部分，看这里：

- [openmm_vggt/models/_mix_decoder_global_base.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/models/_mix_decoder_global_base.py:110)

这个文件定义了很多通用组件，包括：

- `depth_head`
- `point_head`
- `camera_head`
- `track_head`
- `voxel_encoder`
- 点云到 voxel 的编码逻辑
- patch/voxel 对齐的一些公共函数

这里有一个容易误解的点：

配置里虽然经常写 `enable_point=False`，但这通常只是表示“不启用 point prediction head”，不代表模型完全不使用点云。

在这个基类里仍然能看到：

- `voxel_encoder`
- `points`
- `point_mask`
- `lidar_to_world`

这些都在参与早期融合或几何建模。

所以如果你要理解“原模型主干到底怎么吃点云”，一定要看这个基类。

## 5. 新增的 Occupancy Head

真正负责输出体素语义占据结果的模块在这里：

- [openmm_vggt/heads/occupancy_head.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/heads/occupancy_head.py:10)

这是 occupancy 任务里最重要的新模块。

它的大致流程是：

1. 取最后时刻双目图像对应的 patch token
2. 取最后时刻预测的 depth map
3. 用相机内参把 patch 中心点反投影成 3D 射线
4. 用 depth 得到相机坐标下的 3D 点
5. 再通过 `camera_to_world` 和 `lidar_to_world` 变换到 lidar voxel 网格
6. 把 token 聚合到 3D voxel volume
7. 用 `Conv3d` 解码出每个 voxel 的类别 logits

关键上，这个 head 并不是一个简单的 MLP，而是：

- 先把 image token 投影到 voxel
- 再构造 3D volume
- 再走 3D 卷积分类

因此它就是从 2D 图像特征转成 3D occupancy 预测的桥梁。

## 6. Occupancy 训练脚本

对应的训练逻辑在这里：

- [tools/occupancy_train.py](/home/dataset-local/lr/code/openmm_vggt/tools/occupancy_train.py:342)

这里重点看 `compute_losses`。

和原 depth 训练不同，这里支持两个损失通道：

- `depth_weight`
- `occupancy_weight`

occupancy 的损失计算是：

```python
occ_loss = F.cross_entropy(
    occ_logits,
    occ_target,
    ignore_index=occupancy_ignore_index,
    weight=occupancy_class_weights,
)
```

说明 occupancy 微调的监督目标是 voxel 级语义分类，而不是像素深度回归。

同时这里还会统计：

- `occupancy_iou`
- `occupancy_valid_voxels`

所以这个脚本就是 semantic occupancy 微调真正生效的地方。

## 7. Semantic Occupancy 数据集实现

数据集实现文件在这里：

- [openmm_vggt/datasets/kitti_semantic_occ.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/datasets/kitti_semantic_occ.py:211)

这个文件负责：

- 读取 SemanticKITTI 点云和标签
- 对齐 raw KITTI 图像序列
- 构建时序输入
- 生成 `occupancy_target`
- 生成 `occupancy_valid_mask`

尤其值得看的是这里：

- [kitti_semantic_occ.py:381](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/datasets/kitti_semantic_occ.py:381)

这部分会：

- 把点云标签体素化
- 统计占据 voxel 的语义类
- 用 ray casting 补 free-space
- 对未观测区域使用 `ignore_index=255`

这一步是 semantic occupancy 监督成立的基础。

## 8. 原 depth 版本对应配置

作为对照，原始 depth completion 配置在这里：

- [configs/early/kitti_depth_stereo_mix_window_attn_early_ft.py](/home/dataset-local/lr/code/openmm_vggt/configs/early/kitti_depth_stereo_mix_window_attn_early_ft.py:9)

它和 occupancy 版相比，主要区别是：

- 模型类型是 `mix_decoder_global_window_attn_early`
- 数据集是 `KITTIDepthCompletionStereoDataset`
- 训练目标是 `depth_weight = 1.0`
- 没有 `occupancy_head`
- 没有 `occupancy_weight`
- 没有 `occupancy_target`

所以 occupancy 版并不是简单换数据集，而是：

- 换了模型类型
- 新增了 occupancy head
- 新增了 occupancy loss
- 新增了 occupancy 数据集和标签生成逻辑

## 9. 如果你只想抓“增量改动”本身，建议按这个顺序看

### 第一层：配置怎么接进来

- [configs/occupancy/kitti_semantic_occ_mix_window_attn_early_ft.py](/home/dataset-local/lr/code/openmm_vggt/configs/occupancy/kitti_semantic_occ_mix_window_attn_early_ft.py:13)

### 第二层：增量模型怎么继承原模型

- [openmm_vggt/models/fusion_layer/mix_decoder_global_window_attn_early_occ.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/models/fusion_layer/mix_decoder_global_window_attn_early_occ.py:15)

### 第三层：新增 head 怎么做 occupancy 预测

- [openmm_vggt/heads/occupancy_head.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/heads/occupancy_head.py:10)

### 第四层：训练时怎么计算 occupancy loss

- [tools/occupancy_train.py](/home/dataset-local/lr/code/openmm_vggt/tools/occupancy_train.py:342)

### 第五层：标签怎么构造

- [openmm_vggt/datasets/kitti_semantic_occ.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/datasets/kitti_semantic_occ.py:381)

## 10. 最简总结

如果只用一句话概括：

`kitti_semantic_occ_mix_window_attn_early_ft` 的“增量模块”本质上就是：

- 在原 `mix_decoder_global_window_attn_early` 基础上
- 新增 `mix_decoder_global_window_attn_early_occ`
- 再新增一个 [occupancy_head.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/heads/occupancy_head.py:10)
- 并配套新增 [occupancy_train.py](/home/dataset-local/lr/code/openmm_vggt/tools/occupancy_train.py:342) 和 [kitti_semantic_occ.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/datasets/kitti_semantic_occ.py:211)

也就是说，最核心的“新增实现文件”就是这三个：

- [openmm_vggt/models/fusion_layer/mix_decoder_global_window_attn_early_occ.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/models/fusion_layer/mix_decoder_global_window_attn_early_occ.py:15)
- [openmm_vggt/heads/occupancy_head.py](/home/dataset-local/lr/code/openmm_vggt/openmm_vggt/heads/occupancy_head.py:10)
- [tools/occupancy_train.py](/home/dataset-local/lr/code/openmm_vggt/tools/occupancy_train.py:342)

