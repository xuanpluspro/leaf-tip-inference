# 训练完成的计数模型权重

建议文件名：pet_best.pth、p2rloss_best.pth、apgcc_best.pth、sah_best.pth。
这里存放完整模型或本仓库瘦身后的推理 checkpoint，不是 VGG ImageNet 初始化文件。
四模型使用各自训练权重，不能互换。加载完整 checkpoint 不需要另外下载骨干权重。
也可通过 --weights 指向其他绝对路径；模型文件不提交 Git。
