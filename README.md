# Clyde TinyGPT Base Model

This directory contains the base model architecture for the Clyde AI system.

## Capabilities
- Reading comprehension
- Writing generation
- Speaking/TTS
- Listening/STT
- Translation between languages
- Programming in any language
- PowerPoint creation
- Graphic generation
- Tool calling
- Multi-language support

## Structure
- `base_model.json`: Model architecture and capabilities
- `README.md`: This documentation
- `tinygpt_model.py`: Core PyTorch model architecture
- `train.py`: Training script
- `inference.py`: Inference/generation script

## Usage
```python
from tinygpt_model import ClydeModel
model = ClydeModel.from_base('base_model.json')
model.generate("Hello, I want to learn Python")
```