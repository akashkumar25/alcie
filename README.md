# ALCIE - Active Learning for Continual Image Captioning

Code for my Master's thesis (MSc Data Science and AI, Saarland University): continual learning for image captioning, where a captioning model is trained on a sequence of fashion categories and an **episodic memory** of past examples is selected by **active-learning criteria** to limit catastrophic forgetting.

## Problem

Fine-tuning an image-captioning model on a new domain overwrites what it learned before (catastrophic forgetting). Here the stream is six fashion categories, trained in order: accessories, bottoms, dresses, outerwear, shoes, tops. After each category the model is evaluated on every category seen so far, so forgetting is measured per category.

## Method

- **Models:** OFA and BLIP-2, each with its own training and evaluation scripts.
- **Episodic memory:** a bounded buffer (`memory_buffer.py`) holds training examples per category. While training on category *n*, a few buffered examples from earlier categories are replayed every `replay_freq` samples. When a new category starts, a fraction of each earlier category's slots is freed (`--delete_percent`, 1/n in the provided training scripts) to make room.
- **Which examples enter the memory** (`--training_mode`):

| Mode | Selection criterion |
|------|---------------------|
| `random` | Random baseline |
| `uncertainty` | Highest sequence uncertainty of the model (token-level 1 - max probability, averaged over the caption) |
| `diversity` | CLIP image+text features clustered with K-means; samples chosen by distance to cluster centroids |
| `certainty` | Highest CLIP image-text similarity |
| `hybrid` | `alpha * normalised uncertainty + (1 - alpha) * normalised diversity`, with stratified selection (default `alpha = 0.5`) |

Other modes in `train.py`: `basic` (no memory) and `random_no_delete`.

## Evaluation

After training each category, the scripts under `alcie/scripts/evaluation/` caption the test split of every seen category and compute:

- **Lexical metrics:** BLEU-4, ROUGE-L, METEOR
- **Semantic metric:** BERTScore (precision, recall, F1)

`resultsToCSV.py` collects the per-pair JSON files into one table per metric (test category x training stage), and `generate_graphs.py` plots them. Comparing lexical metrics with BERTScore on the same checkpoints is how the forgetting gap between surface wording and meaning is assessed.

## Results

Result files and the fashion dataset are not part of this repository, so no numbers are reported here. To produce them, follow the steps below and run `resultsToCSV.py`. The headline thesis result (lexical vs. semantic forgetting gap) will be added once the result tables are published alongside the code.

## Reproduce

1. Create the environment (see [Installation](#installation)). A CUDA GPU is required for training.
2. Prepare the data in the layout described in [Dataset Structure](#-dataset-structure).
3. Train with a chosen strategy, e.g. `SAMPLING_STRATEGY=hybrid ./alcie/scripts/trainer/train_blip2.sh` (BLIP-2) or `SAMPLING_STRATEGY=uncertainty ./alcie/scripts/trainer/train.sh` (OFA).
4. Evaluate: `python alcie/scripts/evaluation/evaluate_model_blip2.py --sampling_method hybrid` (or `evaluate_model.py` for OFA).
5. Aggregate: `python alcie/scripts/evaluation/resultsToCSV.py` (edit `evaluation_dir` at the top first) and `generate_graphs.py`.

Paths in the example commands below are relative to the `alcie/` source folder; adjust to your checkout.

## Citation

```bibtex
@mastersthesis{kumar2025alcie,
  author = {Kumar, Akash},
  title  = {Active Learning for Continual Image Captioning},
  school = {Saarland University},
  year   = {2025}
}
```

## Quick Start

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

## Strategy details

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

## Running evaluation

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

## License

MIT, see [LICENSE](LICENSE).
