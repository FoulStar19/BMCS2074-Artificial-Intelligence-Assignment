# CNN car detector / CNN 汽车检测器

## Purpose / 目的

This CNN performs the same final task as YOLO: detect `car` bounding boxes in a traffic image. It uses a binary classifier (`car` / `background`) with multi-scale sliding windows and non-maximum suppression (NMS).

这个 CNN 和 YOLO 的最终任务相同：在交通图片中检测 `car` 的边界框。它使用二元分类（`car` / `background`）、多尺度滑动窗口和非极大值抑制（NMS）。

## Original dataset is unchanged / 原始数据集不会被修改

The original YOLO labels and five configured classes remain unchanged:

```text
0 car | 1 truck | 2 bus | 3 motorcycle | 4 bicycle
```

The current annotated samples are all class `0` (`car`). The scripts create a **separate derived folder**, `dataset/cnn_car_background/`; they never edit `dataset/Training`, `dataset/Validation`, or their labels.

目前的已标注样本都是类别 `0`（`car`）。脚本只会建立独立的衍生资料夹 `dataset/cnn_car_background/`；不会修改 `dataset/Training`、`dataset/Validation` 或它们的 labels。

## Run order / 执行顺序

Install packages once:

```powershell
python -m pip install -r requirements.txt
```

Create CNN training data. This produces one car crop and one non-overlapping background crop where possible for each existing car label:

```powershell
python model/cnn/prepare_cnn_dataset.py --clean-output
```

Train the CNN:

```powershell
cd model/cnn
python train_cnn.py --epochs 20 --batch-size 32
```

Evaluate the cropped-image classifier:

```powershell
python evaluate.py
```

Evaluate CNN as a full-frame detector against the same validation YOLO labels used by YOLO:

```powershell
python evaluate_detector.py
```

For a quick initial CPU test, use only 20 validation images:

```powershell
python evaluate_detector.py --max-images 20
```

## Outputs / 输出

- `saved_model/best_car_detector.pth` - trained CNN checkpoint
- `saved_model/training_curves.png` - CNN training graph
- `saved_model/confusion_matrix.png` - binary crop-classification result
- `saved_model/detection_metrics.json` - detection Precision, Recall, F1, mean IoU, mAP@50, and FPS

## Fair comparison / 公平比较

Compare YOLO and CNN on the same validation images and the same car ground-truth boxes. Report Precision, Recall, F1, mean IoU, mAP@50, and FPS. Do not compare CNN crop-classification accuracy directly with YOLO mAP.

使用相同的 validation images 和相同的 car ground-truth boxes 比较 YOLO 与 CNN。报告 Precision、Recall、F1、mean IoU、mAP@50 和 FPS。不要直接把 CNN 裁剪图片的 classification accuracy 和 YOLO 的 mAP 作比较。
