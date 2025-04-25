# 🎨 Text-to-Image Generation Model

*A state-of-the-art diffusion model for generating high-quality images from text prompts*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow)
![Transformers](https://img.shields.io/badge/Transformers-4.28%2B-green)

## 🌟 Key Features
- **Stable Diffusion v1.5** base model with custom fine-tuning
- **Multi-resolution support** (512x512 to 1024x1024)
- **Specialized adapters** for:
  - Photorealistic portraits
  - Anime/cartoon styles
  - Architectural visualization
- **Prompt guidance** with CLIP similarity scoring

## 🚀 Quick Start

### Installation
git clone https://github.com/Osowomuabe/Text-to-image-model.git
cd Text-to-image-model
pip install -r requirements.txt

**Basic Usage**
from generator import TextToImage

model = TextToImage.load_model("sd-v1.5-finetuned")
image = model.generate(
    prompt="A cyberpunk cityscape at night, raining, neon lights reflecting on wet pavement",
    negative_prompt="blurry, distorted, low quality",
    steps=50,
    guidance_scale=7.5
)
image.save("output.png")

🧠 Model Architecture

![Figure 4](https://github.com/user-attachments/assets/9c767a09-970b-49cc-89c5-afbe941ef847)

📂 Repository Structure

├── models/
│   ├── base/                   # Stable Diffusion checkpoints
│   ├── lora/                   # Custom adapters
│   └── textual_inversion/      # Embedding files
├── configs/
│   ├── inference.yaml          # Generation parameters
│   └── training.yaml           # Fine-tuning configs
├── datasets/                   # Training data
│   ├── prompts.csv
│   └── image_pairs/
└── scripts/
    ├── train.py                # Training script
    └── convert_weights.py      # Model conversion

💻 Hardware Requirements
- Minimum: NVIDIA GPU with 8GB VRAM (GTX 1080/Tesla T4)
- Recommended: RTX 3090/A100 for best performance
