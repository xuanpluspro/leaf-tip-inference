# 训练完成的计数模型权重

按模型分目录，保留原文件名即可：

| 模型 | 权重路径 |
|---|---|
| PET | `pet/best_checkpoint.pth` |
| P2RLoss | `p2rloss/ckpt_epoch_best.pth` |
| APGCC | `apgcc/best_checkpoint.pth` |
| SAH | `sah/best_checkpoint.pth` |

这里存放训练完成的完整 checkpoint 或本仓库瘦身后的推理 checkpoint，不是 VGG ImageNet 初始化文件。四模型权重不能互换，加载完整 checkpoint 不需要另外准备骨干预训练权重。

从仓库根目录运行，例如：

```bash
python -u predict.py --model sah --weights weights/sah/best_checkpoint.pth --input data/ck1 --output outputs/ck1_sah --mode full --trust-checkpoint
```

`--trust-checkpoint` 只用于自己训练、可信的文件。`--model` 决定模型架构，`--weights` 明确指定文件；不会自动搜索或猜测。

同一模型有多个实验时，在模型目录下再按实验名分目录，例如 `sah/ricew3999_ep50_bs40_run1/best_checkpoint.pth`。需要保留最新和最佳模型时，可同时放 `checkpoint.pth` 与 `best_checkpoint.pth`；当前训练仓库的 best 默认按最低验证 MAE 选择，不代表最高 F1。

瘦身后可保存为对应模型目录中的 `inference.pth`，推理时显式改 `--weights` 即可。也支持外部绝对路径和既有文件名，不自动移动、覆盖或删除权重。权重及实验文件默认不提交 Git；目录中的说明文件会保留。
