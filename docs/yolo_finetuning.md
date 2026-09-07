# Fine-tuning the YOLOv11 Layout Model

Commands used to fine-tune the document-layout model on annotated research
papers (dataset and runs live on the external training drive).

## Train

```bash
yolo detect train \
  data=/media/mudit/DarkDwine1/ResearchPapersYOLO_FT/training_dataset/round_final/data.yaml \
  model=/media/mudit/DarkDwine1/ResearchPapersYOLO_FT/models/base_model/yolo11n_doc_layout.pt \
  epochs=150 \
  imgsz=1024 \
  batch=5 \
  patience=45 \
  project=/media/mudit/DarkDwine1/ResearchPapersYOLO_FT/models \
  name=yolo11n_doc_layout_imgsz_1024
```

Base model: `Armaggheddon/yolo11-document-layout` (nano). The deployed run
(`yolo11n_doc_layout_imgsz_1024`) was trained at **imgsz 1024** — inference in
`parse_manager` must use the same size (`YOLO_IMGSZ` in
`parse_manager/config.py`). Classes include the standard DocLayNet 11 plus a
fine-tuned `Authors` class.

## Dataset housekeeping

```bash
# Segregate images and labels after annotation export
mv *.jpg *.png ../images/ 2>/dev/null
mv *.txt labels/
```

## Export to ONNX

```bash
yolo export \
  model=.../weights/best.pt \
  format=onnx dynamic=False opset=19
```

## Deploy

Copy the run directory into the repo (git-ignored):

```bash
cp -r /media/mudit/DarkDwine1/ResearchPapersYOLO_FT/models/yolo11n_doc_layout_imgsz_1024 models/
```

`parse_manager` picks up `models/yolo11n_doc_layout_imgsz_1024/weights/best.pt`
automatically.
