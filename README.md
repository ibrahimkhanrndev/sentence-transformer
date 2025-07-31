# Multi-Task Learning with Transformers

A PyTorch implementation of multi-task learning using transformer architecture for natural language processing tasks including sentence classification and Named Entity Recognition (NER).

## 🚀 Features

- **Sentence Transformer**: Base transformer model for generating sentence embeddings
- **Multi-Task Learning**: Simultaneous training on classification and NER tasks
- **Transfer Learning**: Support for pre-trained model initialization and gradual unfreezing
- **Flexible Architecture**: Configurable transformer parameters and task-specific heads
- **Training Infrastructure**: Complete training pipeline with evaluation metrics

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Training](#training)
- [Transfer Learning](#transfer-learning)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- NumPy
- tqdm
- transformers (for transfer learning)

### Setup

```bash
git clone <repository-url>
cd multitask-transformer
pip install torch numpy tqdm transformers
```

## 🚀 Quick Start

### Basic Sentence Transformer

```python
from sentence_transformer import SentenceTransformer, create_padding_mask
import torch

# Initialize model
model = SentenceTransformer(vocab_size=4096)

# Create input
input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])
padding_mask = create_padding_mask(input_ids)

# Get sentence embeddings
embeddings = model(input_ids, src_padding_mask=padding_mask)
print(f"Embeddings shape: {embeddings.shape}")  # [batch_size, d_model]
```

### Multi-Task Learning

```python
from multitask_learning import MultitaskTransformer, MultitaskLoss

# Initialize multi-task model
model = MultitaskTransformer(
    vocab_size=4096,
    num_classes=3,      # Classification classes
    num_ner_tags=5      # NER tags
)

# Classification task
outputs = model(input_ids, task='classification', src_padding_mask=padding_mask)
print(f"Classification logits: {outputs['logits'].shape}")  # [batch_size, num_classes]

# NER task
outputs = model(input_ids, task='ner', src_padding_mask=padding_mask)
print(f"NER logits: {outputs['logits'].shape}")  # [batch_size, seq_len, num_ner_tags]
```

## 🏗️ Architecture

### Core Components

1. **SentenceTransformer** (`sentence_transformer.py`)
   - Token embedding layer
   - Positional encoding
   - Multi-head attention transformer encoder
   - Global average pooling for sentence representations

2. **MultitaskTransformer** (`multitask_learning.py`)
   - Inherits from SentenceTransformer
   - Task-specific classification heads
   - Unified forward pass for multiple tasks

3. **Training Infrastructure** (`trainer.py`)
   - Data loading and batching
   - Multi-task training loop
   - Evaluation metrics
   - Learning rate scheduling

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Embedding dimension |
| `nhead` | 8 | Number of attention heads |
| `num_encoder_layers` | 6 | Number of transformer layers |
| `dim_feedforward` | 2048 | Feedforward network dimension |
| `dropout` | 0.1 | Dropout probability |
| `max_seq_length` | 512 | Maximum sequence length |

## 📝 Usage Examples

### 1. Sentence Embeddings
See `01_sentence_transformer_demo.ipynb` for detailed examples of:
- Basic embedding generation
- Handling variable-length sequences
- Padding mask usage
- Positional encoding verification

### 2. Multi-Task Learning
See `02_multitask_learning_demo.ipynb` for:
- Classification and NER task examples
- Loss computation for multiple tasks
- Model output interpretation

### 3. Transfer Learning
See `03_transfer_learning_strategies.ipynb` for:
- Parameter freezing strategies
- Gradual unfreezing approaches
- Pre-trained model integration

## 🎯 Training

### Basic Training

```python
from trainer import train_model, MultitaskDataset

# Prepare datasets
train_dataset = MultitaskDataset(texts, classification_labels, ner_labels, tokenizer)
val_dataset = MultitaskDataset(val_texts, val_classification_labels, val_ner_labels, tokenizer)

# Train model
history = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=10,
    learning_rate=2e-5
)
```

### Training Features

- **Task Sampling**: Randomly samples between classification and NER tasks
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Linear warmup and decay
- **Evaluation Metrics**: Accuracy for both tasks, task-specific losses

## 🔄 Transfer Learning

### Freezing Strategies

1. **Freeze Entire Network**: Use pre-trained representations
2. **Freeze Backbone**: Train only task-specific heads
3. **Gradual Unfreezing**: Progressively unfreeze layers
4. **Selective Freezing**: Freeze specific task heads

### Example: Backbone Freezing

```python
# Freeze transformer backbone
for param in model.embedding.parameters():
    param.requires_grad = False
for param in model.transformer_encoder.parameters():
    param.requires_grad = False

# Keep task heads trainable
for param in model.classification_head.parameters():
    param.requires_grad = True
for param in model.ner_head.parameters():
    param.requires_grad = True
```

## 📁 Project Structure

```
multitask-transformer/
├── sentence_transformer.py           # Base transformer implementation
├── multitask_learning.py            # Multi-task model and heads
├── trainer.py                       # Training infrastructure
├── 01_sentence_transformer_demo.ipynb    # Basic transformer demos
├── 02_multitask_learning_demo.ipynb      # Multi-task examples
├── 03_transfer_learning_strategies.ipynb # Transfer learning guide
└── README.md                        # This file
```

## 🔧 Model Configuration

### Customizing Architecture

```python
# Custom model configuration
model = MultitaskTransformer(
    vocab_size=10000,
    num_classes=5,
    num_ner_tags=9,
    d_model=768,           # Larger embedding dimension
    nhead=12,              # More attention heads
    num_encoder_layers=12, # Deeper network
    dropout=0.1
)
```

### Task-Specific Heads

- **Classification Head**: Dense → Dropout → Classifier
- **NER Head**: Dense → Dropout → Token Classifier
- Both use GELU activation and configurable dropout

## 📊 Performance Considerations

### Memory Optimization
- Use gradient checkpointing for large models
- Implement gradient accumulation for large batches
- Consider mixed precision training

### Training Tips
- Start with lower learning rates (1e-5 to 5e-5)
- Use warmup for stable training
- Monitor both task losses during training
- Consider task-specific loss weighting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original Transformer architecture from "Attention Is All You Need"
- PyTorch team for the excellent deep learning framework
- Hugging Face for transformer implementations and pre-trained models

## 📚 References

- Vaswani, A., et al. (2017). Attention is all you need. NIPS.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Liu, X., et al. (2019). Multi-Task Deep Neural Networks for Natural Language Understanding. 