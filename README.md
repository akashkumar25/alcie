# ALCIE - Adaptive Learning for Continual Image Caption Enhancement

A comprehensive framework for continual learning in image captioning with advanced memory replay strategies, supporting both **OFA** and **BLIP-2** models for fashion image captioning tasks.

## 🎯 Overview

ALCIE addresses the challenge of catastrophic forgetting in continual learning for image captioning by implementing sophisticated memory management and replay strategies. The system trains models sequentially on different fashion categories while maintaining performance on previously learned categories.

## 🏗️ Repository Structure

```
alcie/
├── scripts/
│   ├── data_processing/           # Data handling components
│   │   ├── __init__.py
│   │   ├── argument.py           # Training arguments for OFA/BLIP2
│   │   ├── datacollator.py       # Data collators for both models
│   │   └── dataset.py            # Dataset classes for OFA/BLIP2
│   ├── memory_managements/       # Memory replay strategies
│   │   ├── __init__.py
│   │   ├── diversity_sampling.py    # Diversity-based sampling
│   │   ├── random_sampling.py       # Random sampling baseline
│   │   ├── certainty_sampling.py    # Certainty-based sampling
│   │   ├── uncertainity_sampling.py # Uncertainty-based sampling
│   │   ├── hybrid_sampling.py       # Hybrid sampling strategy
│   │   └── memory_buffer.py         # Base memory buffer
│   ├── trainer/                  # Training scripts and configurations
│   │   ├── train.py             # OFA training script
│   │   ├── train.sh             # OFA training shell script
│   │   ├── train_blip2.py       # BLIP2 training script
│   │   ├── train_blip2.sh       # BLIP2 training shell script
│   │   ├── train_ofa.json       # OFA training configuration
│   │   ├── train_blip2.json     # BLIP2 training configuration
│   │   └── *_template.json      # Configuration templates
│   └── evaluation/              # Evaluation and analysis
│       ├── evaluate_model.py    # OFA model evaluation
│       ├── evaluate_model_blip2.py  # BLIP2 model evaluation
│       ├── generate_graphs.py   # Performance visualization
│       └── resultsToCSV.py      # Results conversion
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU
- PyTorch 1.12+
- Transformers library
- Additional dependencies in requirements files

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/alcie.git
cd alcie
```

2. **Set up environments:**

For **OFA**:
```bash
# Create conda environment
conda create -n alcie_ofa python=3.8
conda activate alcie_ofa

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install transformers==4.21.0 tokenizers datasets accelerate
pip install evaluate scikit-learn numpy pandas Pillow tqdm
pip install loguru clip-by-openai opencv-python fairseq
pip install omegaconf hydra-core tensorboard wandb
```

For **BLIP-2**:
```bash
# Create conda environment  
conda create -n alcie_blip2 python=3.9
conda activate alcie_blip2

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install transformers==4.35.0 tokenizers datasets accelerate
pip install evaluate scikit-learn numpy pandas Pillow tqdm
pip install loguru clip-by-openai opencv-python tensorboard wandb
pip install peft bitsandbytes sentencepiece protobuf
```

## 📊 Dataset Structure

The system expects fashion image captioning data organized by clusters:

```
cluster_facade_training_combined/
├── accessories/
│   ├── train_caption.jsonl    # Training captions
│   ├── train_image.tsv        # Base64 encoded images
│   ├── test_caption.jsonl     # Test captions
│   └── test_image.tsv         # Test images
├── bottoms/
├── dresses/
├── outerwear/
├── shoes/
└── tops/
```

### Data Format

**Caption File (JSONL)**:
```json
{"image_id": "12345", "text": "A red cotton t-shirt with short sleeves"}
{"image_id": "12346", "text": "Blue denim jeans with straight cut"}
```

**Image File (TSV)**:
```
image_id	base64_image_data
12345	/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcU...
```

## 🧠 Memory Management Strategies

### 1. Random Sampling (`random_sampling.py`)
- **Strategy**: Random selection from training batches
- **Use Case**: Baseline comparison
- **Memory**: Stores random samples with equal probability

### 2. Diversity Sampling (`diversity_sampling.py`)
- **Strategy**: CLIP-based feature clustering with K-means
- **Use Case**: Maximize sample diversity in memory buffer
- **Memory**: Stores samples farthest from cluster centroids

### 3. Uncertainty Sampling (`uncertainity_sampling.py`)
- **Strategy**: Model prediction confidence analysis
- **Use Case**: Focus on challenging samples
- **Memory**: Stores samples with highest prediction uncertainty

### 4. Certainty Sampling (`certainty_sampling.py`)
- **Strategy**: CLIP similarity-based selection
- **Use Case**: Store high-confidence aligned samples
- **Memory**: Stores samples with highest image-text similarity

### 5. Hybrid Sampling (`hybrid_sampling.py`)
- **Strategy**: Combines uncertainty and diversity with stratification
- **Use Case**: Balanced representation across uncertainty levels
- **Memory**: Weighted combination of uncertainty and diversity scores

## 🎓 Training

### OFA Training

```bash
# Activate OFA environment
conda activate alcie_ofa

# Set sampling strategy and run training
export SAMPLING_STRATEGY=uncertainty
./scripts/trainer/train.sh
```

**Manual training for specific cluster:**
```bash
python scripts/trainer/train.py \
    --train_args_file scripts/trainer/train_ofa.json \
    --memory_file memory/uncertainty_sampling/memory_buffer.pkl \
    --current_cluster 2 \
    --total_capacity 10000 \
    --delete_percent 0.5 \
    --training_mode uncertainty \
    --use_memory_replay
```

### BLIP-2 Training

```bash
# Activate BLIP-2 environment
conda activate alcie_blip2

# Set sampling strategy and run training
export SAMPLING_STRATEGY=hybrid
./scripts/trainer/train_blip2.sh
```

**Manual training for specific cluster:**
```bash
python scripts/trainer/train_blip2.py \
    --train_args_file scripts/trainer/train_blip2.json \
    --memory_file memory/blip2/hybrid_sampling/memory_buffer.pkl \
    --current_cluster 3 \
    --total_capacity 10000 \
    --delete_percent 0.33 \
    --training_mode hybrid \
    --use_memory_replay
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--training_mode` | Sampling strategy | `random` |
| `--current_cluster` | Current cluster number (1-6) | `1` |
| `--total_capacity` | Memory buffer capacity | `10000` |
| `--delete_percent` | Memory deletion percentage | `0.0` |
| `--use_memory_replay` | Enable memory replay | `False` |

## 📈 Evaluation

### Evaluate OFA Models

```bash
python scripts/evaluation/evaluate_model.py --sampling_method uncertainty
```

### Evaluate BLIP-2 Models

```bash
python scripts/evaluation/evaluate_model_blip2.py --sampling_method hybrid
```

### Generate Performance Graphs

```bash
python scripts/evaluation/generate_graphs.py
```

### Convert Results to CSV

```bash
python scripts/evaluation/resultsToCSV.py
```

### Evaluation Metrics

The system evaluates models using:
- **BLEU-4**: N-gram overlap precision
- **ROUGE-L**: Longest common subsequence
- **METEOR**: Semantic similarity with synonyms
- **BERTScore**: Contextual embedding similarity

## 🔧 Configuration

### OFA Configuration (`train_ofa_template.json`)

```json
{
    "num_train_epochs": 20,
    "per_device_train_batch_size": 64,
    "learning_rate": 5e-05,
    "max_seq_length": 512,
    "freeze_encoder": false,
    "freeze_word_embed": false
}
```

### BLIP-2 Configuration (`train_blip2_template.json`)

```json
{
    "num_train_epochs": 20,
    "per_device_train_batch_size": 32,
    "learning_rate": 5e-05,
    "max_seq_length": 128,
    "freeze_encoder": true,
    "freeze_qformer": false
}
```

## 📋 Cluster Training Sequence

The training follows this sequence:
1. **Accessories** (cluster 1)
2. **Bottoms** (cluster 2) 
3. **Dresses** (cluster 3)
4. **Outerwear** (cluster 4)
5. **Shoes** (cluster 5)
6. **Tops** (cluster 6)

Each cluster builds upon the previous model checkpoint while managing memory to prevent catastrophic forgetting.

## 🎯 Key Features

### 🔄 Continual Learning
- Sequential training across fashion categories
- Checkpoint-based model progression
- Memory replay to prevent forgetting

### 🧠 Advanced Memory Management
- Multiple sampling strategies
- Dynamic memory capacity management
- Score-based sample selection and deletion

### 📊 Comprehensive Evaluation
- Multiple evaluation metrics
- Cross-cluster performance analysis
- Detailed performance visualization

### 🛠️ Flexible Architecture
- Support for both OFA and BLIP-2 models
- Modular memory management strategies
- Configurable training parameters

## 📁 Output Structure

```
fine_tuned_models/
├── [sampling_strategy]/
│   └── OFA_trained_model_memory/
│       ├── accessories/checkpoint-best/
│       ├── bottoms/checkpoint-best/
│       └── ...
└── blip2/
    └── [sampling_strategy]/
        └── BLIP2_trained_model_memory/
            ├── accessories/checkpoint-best/
            └── ...

evaluation/
├── [sampling_strategy]/
│   ├── evaluation_results_accessories_on_accessories.json
│   └── ...
└── blip2/
    └── [sampling_strategy]/
        └── evaluation_results_*.json

logs/
├── training_[sampling_strategy].log
└── memory_tracking_[sampling_strategy].log
```

## 🔍 Memory Buffer Analysis

Each memory buffer maintains detailed logs:
- Sample addition/deletion events
- Memory capacity changes
- Cluster-wise sample distribution
- Scoring metrics for each sample

View memory logs:
```bash
tail -f logs/memory_tracking_uncertainty.log
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Memory Error**
   - Reduce batch size in config files
   - Reduce memory buffer capacity
   - Use gradient checkpointing

2. **Memory Buffer Full**
   - Increase total capacity
   - Adjust deletion percentages
   - Check memory cleanup logic

3. **Missing Checkpoints**
   - Verify output directory paths
   - Check training completion logs
   - Ensure sufficient disk space

### Environment Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify transformers version
python -c "import transformers; print(transformers.__version__)"

# Check CLIP installation
python -c "import clip; print('CLIP loaded successfully')"
```

## 📚 Research Background

This implementation supports research in:
- **Continual Learning**: Sequential task learning without forgetting
- **Memory Replay**: Strategic sample selection and storage
- **Multimodal Learning**: Vision-language model adaptation
- **Fashion AI**: Domain-specific image captioning

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-sampling-strategy`)
3. Implement changes following existing patterns
4. Add comprehensive tests and documentation
5. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OFA Team**: Original OFA model architecture
- **Salesforce**: BLIP-2 model and research
- **Hugging Face**: Transformers library and model hub
- **OpenAI**: CLIP model for multimodal embeddings
- **Fashion Dataset Contributors**: Curated fashion image datasets

## 📞 Contact

For questions and support:
- Create an issue in this repository
- Check existing documentation and logs
- Review configuration templates for reference

---

**Happy Training! 🚀**
