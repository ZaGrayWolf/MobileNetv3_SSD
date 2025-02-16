# MobileNetv3_SSD

## Custom SSDLite Object Detection

A PyTorch implementation of a custom SSDLite (Single Shot MultiBox Detector Lite) model for efficient object detection, built on top of MobileNetV3 backbone. This implementation is optimized for resource-constrained environments while maintaining good detection performance.

![SSD Architecture](https://miro.medium.com/max/1400/1*0pMP3aHvnGuko54VJlqm2Q.png)

## Features

- Custom SSDLite architecture with MobileNetV3 backbone
- Modified detection head for improved performance
- VOC dataset format support
- Training and validation pipeline
- Data augmentation
- Efficient batch processing
- Custom loss computation
- Non-Maximum Suppression (NMS) for post-processing

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
Pillow>=8.0.0
tqdm>=4.60.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/custom-ssdlite-detection.git
cd custom-ssdlite-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The model expects data in PASCAL VOC format with the following structure:

```
dataset/
├── train/
│   ├── images/
│   └── annotations/
└── val/
    ├── images/
    └── annotations/
```

Each annotation file should be in XML format containing:
- Object class names
- Bounding box coordinates (xmin, ymin, xmax, ymax)

## Model Architecture

The model consists of several key components:

1. **Backbone**: MobileNetV3-Large for feature extraction
2. **Feature Pyramid**: Additional convolution layers for multi-scale feature detection
3. **Custom Detection Head**: Separate classification and regression branches
4. **Custom Transform Module**: Handles image preprocessing and resizing
5. **Loss Computation**: Combined classification and regression loss

## Training

To train the model:

```python
python main.py
```

Default training parameters:
- Epochs: 50
- Batch size: 4
- Learning rate: 1e-4
- Input size: 320x320
- Optimizer: Adam

## Data Augmentation

The training pipeline includes several augmentation techniques:
- Random horizontal flip
- Color jittering (brightness, contrast, saturation, hue)
- Normalization
- Resize to 320x320

## Performance Optimization

The implementation includes several optimizations:
- Custom collate function for efficient batch processing
- Pin memory for faster data transfer to GPU
- Error handling for robust training
- Best model checkpointing

## Model Saving

The best model (based on validation loss) is automatically saved during training. The checkpoint includes:
- Model state dict
- Optimizer state dict
- Training metrics
- Epoch information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{custom-ssdlite-detection,
  author = {Your Name},
  title = {Custom SSDLite Object Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/custom-ssdlite-detection}
}
```

## Acknowledgments

- SSD paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- MobileNetV3 paper: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
