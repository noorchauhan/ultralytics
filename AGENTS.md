# Ultralytics YOLO - AI Agent Guide

This document provides essential information for AI coding agents working on the Ultralytics YOLO project.

## Project Overview

Ultralytics YOLO is a state-of-the-art (SOTA) computer vision framework that provides:

- **Object Detection** - Real-time object detection with bounding boxes
- **Instance Segmentation** - Pixel-level object segmentation
- **Image Classification** - Multi-class image classification
- **Pose Estimation** - Human keypoint detection
- **Oriented Bounding Boxes (OBB)** - Rotated bounding boxes for aerial/satellite imagery
- **Object Tracking** - Multi-object tracking across video frames

The framework supports multiple YOLO model versions including YOLOv3, YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, and YOLO26.

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch >= 1.8.0, TorchVision |
| **Build System** | setuptools, wheel |
| **Testing** | pytest, pytest-cov |
| **Documentation** | MkDocs with Material theme |
| **Linting/Formatting** | Ruff (preferred), YAPF, isort |
| **Docker** | Multi-platform (AMD64, ARM64, Jetson) |

## Project Structure

```
ultralytics/
├── __init__.py           # Package entry point, version info, lazy imports
├── cfg/                  # Configuration files
│   ├── __init__.py       # CLI entrypoint, configuration parsing
│   ├── default.yaml      # Default training/prediction hyperparameters
│   ├── datasets/         # Dataset configuration files (COCO, ImageNet, etc.)
│   ├── models/           # Model architecture YAMLs (YOLOv8, YOLO11, RT-DETR, etc.)
│   └── trackers/         # Tracker configurations (BotSort, ByteTrack)
├── data/                 # Data loading and preprocessing
│   ├── augment.py        # Data augmentation transforms
│   ├── base.py           # Base dataset classes
│   ├── build.py          # DataLoader builders
│   ├── converter.py      # Dataset format converters
│   ├── dataset.py        # Dataset implementations
│   └── loaders.py        # Image/video/stream loaders
├── engine/               # Core training/validation/prediction engine
│   ├── exporter.py       # Model export to various formats
│   ├── model.py          # Base Model class (training, validation, prediction)
│   ├── predictor.py      # Base predictor for inference
│   ├── results.py        # Result containers
│   ├── trainer.py        # Base trainer
│   ├── tuner.py          # Hyperparameter tuning
│   └── validator.py      # Base validator
├── hub/                  # Ultralytics HUB integration
│   ├── auth.py           # Authentication
│   └── session.py        # Training sessions
├── models/               # Model implementations
│   ├── fastsam/          # FastSAM model
│   ├── nas/              # Neural Architecture Search models
│   ├── rtdetr/           # RT-DETR transformer-based detector
│   ├── sam/              # Segment Anything Model (SAM)
│   └── yolo/             # YOLO models (detect, segment, classify, pose, obb, world)
├── nn/                   # Neural network modules
│   ├── modules/          # Building blocks (Conv, Bottleneck, Attention, etc.)
│   ├── tasks.py          # Task-specific model definitions
│   └── autobackend.py    # Auto backend selection for inference
├── optim/                # Optimizers (including Muon)
├── solutions/            # Ready-to-use solutions
│   ├── ai_gym.py         # Workout monitoring
│   ├── heatmap.py        # Heatmap generation
│   ├── object_counter.py # Object counting
│   ├── speed_estimation.py
│   └── streamlit_inference.py
├── trackers/             # Object tracking
│   ├── basetrack.py      # Base tracker
│   ├── bot_sort.py       # BotSort tracker
│   └── byte_tracker.py   # ByteTrack tracker
└── utils/                # Utility functions
    ├── callbacks/        # Training callbacks (ClearML, Comet, W&B, etc.)
    ├── export/           # Export utilities (TensorFlow, OpenVINO, etc.)
    ├── benchmarks.py     # Model benchmarking
    ├── checks.py         # Environment checks
    ├── loss.py           # Loss functions
    ├── metrics.py        # Evaluation metrics
    ├── torch_utils.py    # PyTorch utilities
    └── plotting.py       # Visualization utilities

tests/                    # Test suite
├── conftest.py           # Pytest configuration and fixtures
├── test_cli.py           # CLI tests
├── test_cuda.py          # GPU/CUDA tests
├── test_engine.py        # Engine tests
├── test_exports.py       # Model export tests
├── test_integrations.py  # Integration tests
├── test_python.py        # Python API tests
└── test_solutions.py     # Solutions tests

docs/                     # Documentation
├── en/                   # English documentation
├── overrides/            # MkDocs theme overrides
└── ...

examples/                 # Example notebooks and projects
├── tutorial.ipynb
├── object_counting.ipynb
├── object_tracking.ipynb
├── heatmaps.ipynb
├── YOLOv8-ONNXRuntime/
├── YOLOv8-CPP-Inference/
└── ...

docker/                   # Dockerfiles
├── Dockerfile            # Main CUDA-enabled image
├── Dockerfile-python     # Python-based image
├── Dockerfile-arm64      # ARM64/Apple Silicon
├── Dockerfile-cpu        # CPU-only
├── Dockerfile-jetson-*   # NVIDIA Jetson
└── ...
```

## Installation & Setup

### Development Installation

```bash
# Clone the repository
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# Install in editable mode with all dev dependencies
pip install -e ".[dev,export,solutions]"

# Or use uv (faster, used in CI)
uv pip install -e ".[dev,export,solutions]"
```

### Optional Dependencies

| Group | Purpose |
|-------|---------|
| `dev` | pytest, coverage, mkdocs plugin |
| `export` | ONNX, TensorRT, CoreML, TensorFlow, OpenVINO |
| `solutions` | Streamlit, Flask, Shapely |
| `logging` | Weights & Biases, TensorBoard, MLflow |
| `extra` | Albumentations, faster-coco-eval |
| `typing` | Type stubs for better IDE support |

## Build and Test Commands

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=ultralytics/ --cov-report=xml tests/

# Run slow tests (marked with @pytest.mark.slow)
pytest --slow tests/

# Run specific test file
pytest tests/test_python.py

# Run specific test
pytest tests/test_python.py::test_model_forward
```

### Code Quality

```bash
# Format code with Ruff (preferred)
ruff check --fix .
ruff format .

# Or use YAPF (legacy)
yapf -i -r ultralytics/

# Sort imports
isort ultralytics/

# Check code spelling
codespell
```

### CLI Usage

```bash
# Check environment
yolo checks

# Train a model
yolo train model=yolo26n.pt data=coco8.yaml epochs=100 imgsz=640

# Validate
yolo val model=yolo26n.pt data=coco8.yaml

# Predict
yolo predict model=yolo26n.pt source='path/to/image.jpg'

# Export
yolo export model=yolo26n.pt format=onnx

# Benchmark
yolo benchmark model=yolo26n.pt

# View configuration
yolo cfg
```

### Documentation

```bash
# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

## Code Style Guidelines

### Python Code Style

- **Line length**: 120 characters (configured in `pyproject.toml`)
- **Formatter**: Ruff (preferred) or YAPF
- **Import style**: isort with `multi_line_output = 0`

### Docstrings

Use **Google-style docstrings** with specific formatting requirements:

```python
def example_function(arg1: int, arg2: int = 4) -> bool:
    """Example function demonstrating Google-style docstrings.

    Args:
        arg1: The first argument.
        arg2: The second argument, with a default value of 4.

    Returns:
        True if successful, False otherwise.

    Examples:
        >>> result = example_function(1, 2)  # returns False
    """
    if arg1 == arg2:
        return True
    return False
```

**Key requirements:**
- Input and output types must be enclosed in parentheses: `(bool)`, `(int)`
- Include `Examples` section with doctests when applicable
- Single-line docstrings for simple functions: `"""Brief description."""`

### File Headers

All Python files must include the license header:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `DetectionModel`, `BaseTrainer`)
- **Functions/Methods**: `snake_case` (e.g., `predict`, `train_one_epoch`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CFG`, `TASKS`)
- **Private members**: `_leading_underscore` (e.g., `_load`, `_new`)

### Type Hints

Use type hints for function signatures:

```python
def train(self, data: str | None = None, **kwargs) -> None:
    ...
```

## Testing Strategy

### Test Categories

1. **Unit Tests** (`test_python.py`) - Test individual components
2. **CLI Tests** (`test_cli.py`) - Test command-line interface
3. **Export Tests** (`test_exports.py`) - Test model export formats
4. **Engine Tests** (`test_engine.py`) - Test training/validation engine
5. **CUDA Tests** (`test_cuda.py`) - GPU-specific tests
6. **Integration Tests** (`test_integrations.py`) - Third-party integrations
7. **Solution Tests** (`test_solutions.py`) - Ready-to-use solutions

### Test Configuration

Tests use `conftest.py` for:
- Custom `--slow` flag to skip slow tests
- Automatic seed initialization (`init_seeds()`)
- Cleanup of test artifacts after test session

### Test Constants (in `tests/__init__.py`)

```python
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"
CFG = "yolo26n.yaml"
SOURCE = ASSETS / "bus.jpg"
TASK_MODEL_DATA = [(task, model, data) for task in TASKS]
```

## Architecture Patterns

### Model Architecture

The framework uses a unified `Model` class (`ultralytics/engine/model.py`) that:
- Inherits from `torch.nn.Module`
- Provides common API: `train()`, `val()`, `predict()`, `export()`, `benchmark()`
- Uses lazy loading for model classes via `__getattr__`

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # Load pretrained model
results = model.predict("image.jpg")  # Inference
model.train(data="coco8.yaml", epochs=100)  # Training
```

### Task-Specific Implementations

Each task (detect, segment, classify, pose, obb) has:
- **Model definition** in `ultralytics/models/yolo/{task}/`
- **Trainer** - `train.py`
- **Validator** - `val.py`
- **Predictor** - `predict.py`

### Configuration System

- **Default config**: `ultralytics/cfg/default.yaml`
- **Loading**: `get_cfg()` function in `ultralytics/cfg/__init__.py`
- **Override precedence**: CLI args > custom yaml > default yaml

### Callback System

Training callbacks are defined in `ultralytics/utils/callbacks/`:
- Base callbacks in `base.py`
- Integration callbacks: `wb.py` (Weights & Biases), `comet.py`, `tensorboard.py`, etc.
- Add callbacks: `model.add_callback("on_train_epoch_end", callback_fn)`

## CI/CD Pipeline

GitHub Actions workflows in `.github/workflows/`:

| Workflow | Purpose |
|----------|---------|
| `ci.yml` | Main CI - tests on multiple OS/Python versions, HUB tests, benchmarks |
| `docker.yml` | Build and publish Docker images |
| `docs.yml` | Build and deploy documentation |
| `format.yml` | Code formatting and linting |
| `publish.yml` | Publish to PyPI |
| `links.yml` | Check documentation links |
| `cla.yml` | Contributor License Agreement |

### CI Jobs

1. **HUB** - Test Ultralytics HUB integration
2. **Benchmarks** - Performance benchmarks on different platforms
3. **Tests** - pytest on Ubuntu/macOS/Windows/ARM64
4. **SlowTests** - Extended tests with historical PyTorch versions
5. **GPU** - CUDA-specific tests on GPU runners
6. **RaspberryPi** - ARM tests on Raspberry Pi
7. **NVIDIA_Jetson** - Tests on Jetson hardware
8. **Conda** - Conda package testing

## Security Considerations

- **License**: AGPL-3.0 (open source, copyleft)
- All code changes must pass CLA check
- Enterprise license available for commercial use
- No sensitive data should be committed (model weights are downloaded from releases)

## Common Development Tasks

### Adding a New Model

1. Create model YAML in `ultralytics/cfg/models/`
2. Add model class in `ultralytics/models/`
3. Register in `ultralytics/models/__init__.py`
4. Add tests in appropriate test file

### Adding a New Task

1. Create task directory in `ultralytics/models/yolo/{task}/`
2. Implement `train.py`, `val.py`, `predict.py`
3. Add task to `TASKS` in `ultralytics/cfg/__init__.py`
4. Add dataset mapping in `TASK2DATA`

### Adding a New Export Format

1. Add export logic in `ultralytics/engine/exporter.py`
2. Add format-specific utilities in `ultralytics/utils/export/`
3. Add tests in `tests/test_exports.py`

### Adding a Solution

1. Create solution class in `ultralytics/solutions/`
2. Inherit from `BaseSolution`
3. Register in `SOLUTION_MAP` in `ultralytics/cfg/__init__.py`
4. Add tests in `tests/test_solutions.py`

## Important Notes

- **OMP_NUM_THREADS**: Set to 1 by default in `ultralytics/__init__.py` to reduce CPU utilization
- **Windows compatibility**: Avoid hardcoded paths, use `pathlib.Path`
- **CUDA version compatibility**: Check `torch` version constraints in `pyproject.toml`
- **Model downloads**: Models are automatically downloaded from GitHub releases on first use
- **Caching**: Image caching can be enabled via `cache=True` or `cache='disk'`

## External Resources

- **Documentation**: https://docs.ultralytics.com
- **GitHub**: https://github.com/ultralytics/ultralytics
- **PyPI**: https://pypi.org/project/ultralytics
- **Docker Hub**: https://hub.docker.com/r/ultralytics/ultralytics
- **Discord**: https://discord.com/invite/ultralytics
- **Community Forum**: https://community.ultralytics.com
