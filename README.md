# leaf-tip-inference

统一 PET、P2RLoss、APGCC、SAH 的植物叶尖推理，用 RiceW3999 训练好的权重预测黑麦草。四个模型源码按 UPSTREAM.json 固定版本内置，不需要同时克隆四个训练仓库。

## 默认输出和行为

- GPU 单图整图推理，默认 FP32，与训练仓库验证精度设置一致。不会自动缩放、切块或切换 CPU。
- 默认保存红色预测点 + 数量文字的 JPG 叠加图，文件名 leaf_overlay.jpg，quality=90；保持原始尺寸。
- 同时保存 xy 坐标 TXT/JSON、counts.csv、meta/ 日志和运行清单；全部在 --output 指定目录。
- --mode tile 显式启用滑窗；默认窗口 512、重叠 128；每次处理一块，归属边界剔除重叠区重复预测。不做距离 NMS，以免删除相邻真实叶尖。
- --resume 跳过已完成图片；--overwrite 全部重算并覆盖同名输出，二者互斥。
- 可选 --amp 开启 FP16 autocast；不保证与 FP32 数值或阈值附近计数完全一致。

## 准备环境与文件

GPUFree 已有 torch/torchvision 就保留，不要重复安装。先检查：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('GPU available:', torch.cuda.is_available())"
```

逐个安装辅助依赖（上交镜像）：

```bash
uv pip install --python "$(command -v python)" numpy -i https://mirror.sjtu.edu.cn/pypi/web/simple
uv pip install --python "$(command -v python)" Pillow -i https://mirror.sjtu.edu.cn/pypi/web/simple
uv pip install --python "$(command -v python)" scipy -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

CPU 版 torch 不支持本工具推理；不存在 CPU 自动回退。轻量测试和权重文件瘦身可以用 CPU，它们不运行神经网络。

```bash
cd /root/gpufree-data
git clone https://github.com/xuanpluspro/leaf-tip-inference.git
cd leaf-tip-inference
```

把图片放 data/ck1/，将训练完成的权重放 weights/（也可使用外部绝对路径）。不需要另外准备 vgg16_bn-6c64b313.pth。

| --model | 原始权重来源 | 默认前景筛选 | 补齐倍数 |
|---|---|---|---|
| pet | best_checkpoint.pth 中 model；读取 args 结构配置 | softmax 前景概率 >0.5 | 256 |
| p2rloss | ckpt_epoch_best.pth 中 student | sigmoid(output)>0.5，对应 output>0 | 32 |
| apgcc | best_checkpoint.pth 中 model + model_config | softmax 前景概率 >0.5 | 16 |
| sah | best_checkpoint.pth 中 model + model_config | sigmoid 前景概率 >0.5 | 16 |

不同算法权重不能互换。加载严格校验键名和尺寸，不用 strict=False 掩盖模型不匹配。PET 默认结构为 VGG16-BN、hidden_dim=256、dec_layers=2、FFN=512、nheads=8；checkpoint 有 args 时按保存配置恢复。无 args 的旧文件只能按此明确默认配置构建，并严格检查权重；非本版本架构可能不兼容。

原始训练 checkpoint 可能包含 argparse.Namespace 或 NumPy RNG 等 pickle 对象。默认只安全加载权重对象；失败时给出提示。仅对自己训练、可信的文件显式增加 --trust-checkpoint；不要对未知来源文件使用。下面 PET 示例加此参数用于自己的原始训练文件。

## 开始推理

PET 整图示例（你当前完成训练的路径仍带 bs16，但实际最后使用 batch 8）：

```bash
python -u predict.py --model pet \
  --weights /root/gpufree-data/PET/outputs/RiceW3999/ricew3999_pet_ep50_bs16_run1/best_checkpoint.pth \
  --input data/ck1 --output outputs/ck1_pet --mode full --trust-checkpoint
```

其他模型使用同一入口、独立输出目录：

```bash
python -u predict.py --model p2rloss --weights weights/p2rloss_best.pth --input data/ck1 --output outputs/ck1_p2rloss --mode full --trust-checkpoint
python -u predict.py --model apgcc --weights weights/apgcc_best.pth --input data/ck1 --output outputs/ck1_apgcc --mode full --trust-checkpoint
python -u predict.py --model sah --weights weights/sah_best.pth --input data/ck1 --output outputs/ck1_sah --mode full --trust-checkpoint
```

先在新输出目录加 --max-images 1 做一张图的 GPU 检查；确认权重兼容、效果和显存后，去掉 --max-images，加 --resume 继续整个文件夹。max-images 只限制处理数量，不改变预测设置。

整图 OOM 后显式尝试 --mode tile --tile-size 512 --overlap 128，并使用新输出目录。整图与滑窗上下文不同，结果不保证一致。归属法避免同一坐标区域由两块重复负责，但两块对同一目标定位发生偏移时，仍可能在边界出现漏计/重复；需要看可视化，不能宣称完全无缝。

--threshold 0.5 为四者统一的前景概率阈值。P2RLoss 这里传的是概率阈值，不是旧 infer_sudan.py 的原始 logit 阈值；0.5 对应 logit 0。--p2r-weights teacher 可选择教师，但默认学生才与当前训练验证保持一致。

--radius 4 控制画点大小，不影响计数；--jpg-quality 90 控制磁盘压缩质量；--no-save-points 仅保存 overlay/CSV，不输出逐图 TXT/JSON。支持递归子目录；同目录 leaf.jpg 和 leaf.png 会报输出命名冲突，不生成双扩展名。中文路径采用文件对象读写。

```text
outputs/ck1_pet/
  overlays/523_ck1_BLN_overlay.jpg
  points/523_ck1_BLN.txt
  points/523_ck1_BLN.json
  counts.csv
  meta/console.log
  meta/run.json
```

坐标始终是原图像素 xy，右/下 padding 区预测会删除。TXT 每行 x y；JSON 额外保存前景概率、阈值和数量。counts.csv 每完成一张原子更新，包含数量、原尺寸、推理秒数、保存秒数和 PyTorch 峰值已分配 GiB。耗时包含 CPU 标准化、GPU 推理、筛选和坐标回传，不包含读图/加载模型/保存图片，不等同于纯网络延迟。没有 GT 时不报告 MAE/F1 等精度指标。

## 导入 Labelme / X-AnyLabeling 人工修正

`points/*.json` 是本仓库的预测记录，不是标注软件格式。新增 `export_annotations.py` 将**已经推理完成的结果**转为 Labelme/X-AnyLabeling 可读取的点标注 JSON，不需要重新加载模型或 GPU。

```bash
python export_annotations.py \
  --input data/ck1 \
  --predictions outputs/ck1_pet \
  --output review/ck1_pet \
  --label leaf_tip
```

`--predictions` 指向含有 `meta/run.json` 和 `points/` 的推理结果根目录，不是直接指定 `points/`。四个模型共用这一导出入口，替换结果目录即可。若之前用了 `--no-save-points`，需要先重新推理保存坐标。

导出只读取运行清单中已完成的图片；例如先推理一张，也可以只导出这一张，不要求整批结束。需使用推理时相同的原图，不要拿叠加图作为输入。

```text
review/ck1_pet/
  523_ck1_BLN.jpg
  523_ck1_BLN.json
  子目录/
    另一张图片.png
    另一张图片.json
```

- 图像文件直接复制原始字节，不缩放、不重新压缩；原目录不修改。导出包因此会额外占用一份原图空间。
- JSON 与图片同目录、同主干，`imagePath` 仅保存图片文件名；整个文件夹搬到 Windows 后仍能关联原图。
- `imageData: null`，不把图片再次以 Base64 放进 JSON，避免标签体积膨胀。
- 每个形状是 `shape_type: point`，`points: [[x,y]]`，坐标为原图像素浮点数。默认标签 `leaf_tip` 可用 `--label` 更改。
- 置信度保留在各 shape 的 `score` 字段；手工新增点无需填写 score。GUI 是否显示或保留该扩展字段取决于版本，不影响坐标回转。
- `checked: false` 表示未审核。JSON 的 `version: 5.0.1` 采用常见 Labelme 兼容格式标记，不表示导出脚本依赖该软件版本。
- 默认导出全部已保存预测，包括零预测图片（`shapes: []`），方便人工补点。可显式加 `--min-score 0.7` 进一步过滤，但不能恢复推理时已经被阈值丢弃的点。
- 检查坐标范围、尺寸和记录数量；存在非正常 EXIF 旋转标记时拒绝导出，以免标注软件自动旋转图片造成坐标错位。应在推理前统一图像方向后重跑，不会暗中旋转坐标。

将 `review/ck1_pet/` 整个文件夹下载到本地。在 Labelme 或 X-AnyLabeling 中打开图片文件夹或对应 JSON，删除误检点、拖动偏移点、用 point 工具补充漏点，保存到这个导出包中。存在子目录时逐个打开需要处理的子目录。

**导出目录必须尚不存在，没有覆盖开关。** 重复导出会报错，保护已人工修正的 JSON；需要另一批结果时用新目录。推理入口仍保留原来的 TXT/坐标 JSON/overlay 输出，不直接往原图目录写标签。

### 修正后转回训练点标注

在完成审核的本地导出包上执行（只需 numpy/Pillow，不需要 torch 或标注软件 Python 包）：

```bash
python annotations_to_points.py \
  --input review/ck1_pet \
  --output corrected/ck1_pet/PointAnnotations \
  --label leaf_tip
```

生成同名 `.txt`，每行 `x y`，可用于当前 RiceW3999 风格的数据加载器。目录层级保留；现有训练集若要求平铺，请先统一图片/标注文件主干及目录，不会自动重建训练/验证划分。

回转读取人工保存后的 `shapes`，不依赖旧预测分数或数量，因此删点、移点、补点都会反映到新 TXT。全空标注生成空 TXT；重复坐标、越界坐标、非 point 形状或意外标签会明确报错，不静默删除标注。回转目录也必须是新目录，不修改 JSON、原图或既有训练标注。

程序不能判断你是否逐张完成了人工审核：未经修正的预测仍是伪标签，不应直接当作人工 GT；新标注用于训练与独立测试时，应保持合理划分。

格式依据官方读取逻辑核对：
[Labelme](https://github.com/wkentaro/labelme/blob/main/labelme/_label_file.py)、
[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling/blob/main/anylabeling/views/labeling/label_file.py)、
[X-AnyLabeling point shape](https://github.com/CVHub520/X-AnyLabeling/blob/main/anylabeling/views/labeling/shape.py)。
新增 14 项不运行模型的测试，覆盖导出格式、中文路径、图像字节不变、零点、保护人工修改及模拟删/移/补点后的回转。尚未在桌面 GUI 实际打开验证；建议先导出一张，在你的软件版本中确认显示和保存正常。

## 中断续跑与覆盖

相同命令加 --resume；核对模型、权重 SHA256、输入根目录、源码版本及预测/可视化设置。输入逐图 size/mtime 变化时重新处理，未完成文件重做。不能检测刻意维持 size/mtime 的内容篡改，也不验证被人为修改的输出文件。切换模型、权重、阈值或模式，建议新目录；若明确想覆盖，可用 --overwrite。

--overwrite 不进回收站，不备份；覆盖本次对应同名文件，重建 CSV/运行清单和日志，其他文件不删除。处理期间可能同时存在新旧图片；中断后去掉 --overwrite 改加 --resume 完成。不要并发写同一输出目录。原图和权重不修改。

## 可选：瘦身模型文件

```bash
python slim_checkpoint.py --model pet --weights weights/pet_best.pth --output weights/pet_inference.pth --trust-checkpoint
python slim_checkpoint.py --model p2rloss --weights weights/p2rloss_best.pth --output weights/p2rloss_inference.pth --p2r-weights student --trust-checkpoint
```

APGCC/SAH 同理。仅保留选择的一个模型及构造所需元数据，删除优化器/RNG/未选择的教师等；不转换 FP16，不剪枝，逐张量校验数值/dtype/shape，原文件保留。瘦身文件可直接由本仓库加载，不能用于完整续训，也不承诺可以直接交给其他训练仓库的加载器。

## 测试与适用边界

```bash
python -m unittest discover -s tests -v
python -m compileall -q inference vendor predict.py slim_checkpoint.py
```

测试只使用小数组、模拟预测器和 checkpoint 元数据，不安装 torch、不构造或运行模型。真实四种 checkpoint 加载、CUDA FP32/AMP、3072 整图显存和黑麦草效果需在服务器验证，不能将轻量测试当成精度复现。APGCC 继承适配仓库的第三方复现差异，详见 vendor/apgcc/REPRODUCTION.md。

本工具不自动套用分割仓库的“最大连通域”：点集并没有对应的植物前景掩膜。若需要仅统计主体植株，可另行结合分割掩膜过滤，但目前不默认启用。

来源、许可证及必要适配见 THIRD_PARTY.md。模型加载时不联网下载，原四个训练仓库保持不变。
