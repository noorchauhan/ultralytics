# CLAUDE.md

1. **Minimal**: Simplest solution that works - no over-engineering
2. **Replace > Add**: Modify existing code over adding new
3. **Delete ruthlessly**: Remove unused code completely
4. **Cross-app awareness**: If a feature exists in one app, check if it can be reused or adapted
5. **Code Cleanup**: Every time you add or replace code, you MUST clean up what you replaced. Dead code accumulates quickly and creates confusion, maintenance burden, and technical debt.
6. **Fail fast**: Don't add any try catch and protect method, keep it simple and let it fail fast.

## Project Overview

Ultralytics is a Python library for state-of-the-art computer vision models, primarily YOLO variants. It supports object detection, segmentation, classification, pose estimation, oriented bounding boxes (OBB), and stereo 3D detection tasks.

## Common Commands

### Installation
```bash
pip install -e .                    # Editable install for development
pip install -e ".[dev]"             # With dev dependencies
pip install -e ".[export]"          # With export dependencies
```

### Running Tests
```bash
pytest tests/                       # Run all tests
pytest tests/test_python.py         # Run a specific test file
pytest tests/ -k "test_name"        # Run tests matching pattern
pytest tests/ --slow                # Include slow tests (marked @pytest.mark.slow)
pytest --cov=ultralytics/ tests/    # Run with coverage
```

### CLI Usage
```bash
yolo predict model=yolo11n.pt source=image.jpg
yolo train model=yolo11n.pt data=coco8.yaml epochs=100
yolo val model=yolo11n.pt data=coco8.yaml
yolo export model=yolo11n.pt format=onnx
yolo checks                         # Verify environment setup
```

### Linting/Formatting
The project uses ruff (line-length=120), isort, and yapf. CI runs format checks automatically.

## Architecture

### Directory Structure
```
ultralytics/
├── cfg/              # YAML configs for models, datasets, default args
├── data/             # Data loading, augmentation, dataset classes
├── engine/           # Core train/val/predict pipelines (BaseTrainer, BaseValidator, BasePredictor)
├── hub/              # Ultralytics HUB platform integration
├── models/           # Model implementations organized by architecture
│   └── yolo/         # YOLO variants with task-specific submodules
├── nn/               # Neural network layers and modules
│   ├── tasks.py      # Model classes (DetectionModel, SegmentationModel, etc.)
│   └── modules/      # Building blocks (conv, block, head, transformer)
├── solutions/        # Pre-built CV applications (counting, tracking, etc.)
├── trackers/         # Object tracking (ByteTrack, BoT-SORT)
└── utils/            # Utilities (metrics, loss, plotting, torch_utils)
```

### Task System

The codebase uses a task-based architecture. Each task has its own trainer/validator/predictor:

| Task | Directory | Description |
|------|-----------|-------------|
| `detect` | `models/yolo/detect/` | Object detection |
| `segment` | `models/yolo/segment/` | Instance segmentation |
| `classify` | `models/yolo/classify/` | Image classification |
| `pose` | `models/yolo/pose/` | Keypoint/pose estimation |
| `obb` | `models/yolo/obb/` | Oriented bounding boxes |
| `stereo3ddet` | `models/yolo/stereo3ddet/` | Stereo 3D detection (custom) |

Tasks are registered in `YOLO.task_map` property in `models/yolo/model.py`.

### Key Base Classes

- `engine/model.py` → `Model` - High-level API for train/val/predict/export
- `engine/trainer.py` → `BaseTrainer` - Training loop with hooks for customization
- `engine/validator.py` → `BaseValidator` - Evaluation pipeline
- `engine/predictor.py` → `BasePredictor` - Inference pipeline
- `data/base.py` → `BaseDataset` - Dataset loading interface
- `nn/tasks.py` → `BaseModel`, `DetectionModel`, etc. - Model architectures

### Adding a New Task

1. Create module at `ultralytics/models/yolo/newtask/` with:
   - `__init__.py` exporting trainer/validator/predictor
   - `train.py` extending `BaseTrainer`
   - `val.py` extending `BaseValidator`
   - `predict.py` extending `BasePredictor`
2. Register in `YOLO.task_map` in `models/yolo/model.py`
3. Add model configs in `cfg/models/newtask/`
4. Add task to `TASKS` set in `cfg/__init__.py`

### Stereo 3D Detection (Custom Task)

This repo includes a stereo 3D detection implementation in `models/yolo/stereo3ddet/`:

- 6-channel stereo input (left + right images)
- CenterNet-style architecture with weight-shared backbone
- KITTI AP3D evaluation metrics
- Stereo-specific augmentations in `augment.py`
- Geometric 3D construction in `geometric.py`
- Dense alignment for sub-pixel refinement in `dense_align_optimized.py`

Model config: `cfg/models/26/yolo26-stereo3ddet-siamese.yaml`
Dataset config: `cfg/datasets/kitti-stereo.yaml`

### Configuration

- Model architectures defined in YAML: `cfg/models/`
- Dataset configs in YAML: `cfg/datasets/`
- Default training args: `cfg/default.yaml`
- CLI entrypoint: `ultralytics/cfg/__init__.py:entrypoint()`

## Code Style

- Google-style docstrings required for new functions/classes
- Line length: 120 characters
- Type hints encouraged but not required
- Tests in `tests/` directory, use pytest

## Python API Example

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=100, imgsz=640)
model.val()
results = model("image.jpg")
model.export(format="onnx")
```
