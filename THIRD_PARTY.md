# 模型来源和适配改动

所有来源提交固定记录在 UPSTREAM.json，不自动跟随原仓库更新。

- PET：xuanpluspro/PET，源自 ICCV 2023 Point-Query Quadtree；保留 vendor/pet/LICENSE。源 README 同时写有学术用途说明，使用时也应查阅原项目说明。
- P2RLoss：xuanpluspro/p2rloss，保留模型文件内来源信息，来源仓库见 UPSTREAM.json。本次读取的源目录未提供独立 LICENSE，因此不擅自为其添加 MIT 等授权。
- APGCC：xuanpluspro/apgcc-leaf-counting，包含基于 AaronCIH/APGCC 部分公开代码的第三方训练复现。保留 vendor/apgcc/LICENSE、THIRD_PARTY.md、REPRODUCTION.md，不能把本适配的精度当作原论文的复现保证。
- SAH：xuanpluspro/sah-leaf-counting，基于 kaijiang77/SAHCC 适配；保留 vendor/sah/LICENSE、THIRD_PARTY.md、REPRODUCTION.md。
- 工程组织参考 xuanpluspro/plant-seg-inference：独立 weights/data/outputs、显式整图/滑窗、文件对象读写中文路径、原子替换、resume/overwrite。

适配改动：

1. 隔离为 vendor.pet / vendor.p2rloss / vendor.apgcc / vendor.sah，PET 的绝对 util 导入改为命名空间导入，避免四仓库同名 models/util 冲突。
2. PET 构建骨干时不加载 ImageNet 初始化文件，省略不会进入 PET 模型的 VGG 全连接分类器；完整 checkpoint 仍 strict=True 加载，参数键与计算结构保持匹配。PET 推理概率阈值由 kwargs 传入，默认仍为 0.5。
3. P2RLoss 仅构造 VGG16-BN 卷积/BN 特征层，切分仍为 [0:33] 与 [33:43]，保留 encoders/fuse_layer/decoders 参数命名和前向规则。省略 ImageNet 下载与无用 FC，完整学生/教师权重严格加载。
4. APGCC/SAH 模型源码保持源版本逻辑，构造时不传骨干初始化文件，通过 checkpoint 的 model_config 恢复配置。
5. 统一 ImageNet RGB 标准化、原图 xy 输出、pad 区域过滤。PET 保留 256 补齐规则；P2RLoss 为 32；APGCC/SAH 为 16。推理 checkpoint 不需要训练损失，但部分 PET 源模块仍带有未调用的训练辅助定义。

未引入训练入口，不修改任何原训练仓库，不上传数据或权重。
