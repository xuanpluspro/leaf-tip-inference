# apgcc 权重

将此模型训练完成的 `best_checkpoint.pth` 放在本目录。可以保留原文件名；其他模型的同名权重放各自目录。

多次实验可继续按实验名建立子目录。推理用 `--model apgcc --weights weights/apgcc/best_checkpoint.pth` 明确选择文件。

这里只存放训练后的模型；不需要另放 VGG 初始化权重。权重文件不会提交 Git。详见 [上级说明](../README.md)。
