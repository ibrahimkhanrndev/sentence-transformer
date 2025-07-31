import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from typing import Dict, Optional
from sentence_transformer import (
    SentenceTransformer, 
    create_padding_mask
)

class SentenceClassificationHead(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_classes: int
    ):
        """
        Task-specific head for sentence classification.

        Args:
            d_model: embedding dimension
            num_classes: number of classification classes
        """
        super().__init__()
        
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        # x shape: [batch_size, d_model]
        x = self.dropout(F.gelu(self.dense(x)))
        return self.classifier(x)

class NERHead(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_ner_tags: int
    ):
        """
        Task-specific head for Named Entity Recognition.
        
        Args:
            d_model: embedding dimension
            num_ner_tags: number of NER tags
        """
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_ner_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = self.dropout(F.gelu(self.dense(x)))
        return self.classifier(x)

class MultitaskTransformer(SentenceTransformer):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_ner_tags: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        """
        Initialize the MultitaskTransformer model.

        Args:
            vocab_size: size of vocabulary
            num_classes: number of classification classes
            num_ner_tags: number of NER tags
            d_model: embedding dimension
            nhead: number of attention heads
            num_encoder_layers: number of transformer encoder layers
            dim_feedforward: dimension of feedforward network
            dropout: dropout probability
            max_seq_length: maximum sequence length
        """
        super().__init__(
            vocab_size, 
            d_model, 
            nhead, 
            num_encoder_layers, 
            dim_feedforward, 
            dropout, 
            max_seq_length
        )
        
        # Task-specific heads
        self.classification_head = SentenceClassificationHead(d_model, num_classes)
        self.ner_head = NERHead(d_model, num_ner_tags)
        
    def forward(
        self,
        src: Tensor,
        task: str,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the model.
        Args:
            src: input tensor of shape [batch_size, seq_len]
            task: either 'classification' or 'ner'
            src_mask: mask for self-attention
            src_padding_mask: mask for padding tokens
        Returns:
            Dictionary containing task-specific outputs
        """
        # Embed tokens and apply positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(
            x,
            src_mask,
            src_padding_mask
        )
        
        outputs = {}
        
        if task == 'classification':
            # Global average pooling for classification
            pooled = self.pooling(encoded.transpose(1, 2))
            pooled = pooled.squeeze(-1)
            outputs['logits'] = self.classification_head(pooled)
            
        elif task == 'ner':
            # Token-level predictions for NER
            outputs['logits'] = self.ner_head(encoded)
            
        return outputs

class MultitaskLoss:
    def __init__(self):
        """
        Loss functions for multiple tasks.
        """
        self.classification_loss = nn.CrossEntropyLoss()
        self.ner_loss = nn.CrossEntropyLoss(ignore_index=-100)  # -100 is padding index
        
    def __call__(
        self,
        outputs: Dict[str, Tensor],
        labels: Tensor,
        task: str
    ) -> Tensor:
        """
        Calculate loss for the specified task.
        
        Args:
            outputs: model outputs dictionary
            labels: ground truth labels
            task: either 'classification' or 'ner'
        
        Returns:
            Loss tensor
        """
        if task == 'classification':
            return self.classification_loss(outputs['logits'], labels)
        elif task == 'ner':
            # Reshape logits and labels for NER loss
            logits = outputs['logits'].view(-1, outputs['logits'].size(-1))
            labels = labels.view(-1)
            print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

            assert logits.shape[0] == labels.shape[0], f"Mismatch: logits {logits.shape}, labels {labels.shape}"
            return self.ner_loss(logits, labels)
        else:
            raise ValueError(f"Unknown task: {task}")

# Example usage
def test_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    vocab_size = 1000
    num_classes = 3  # Example: positive, negative, neutral
    num_ner_tags = 5  # Example: O, B-PER, I-PER, B-ORG, I-ORG
    batch_size = 2
    seq_length = 10
    
    # Initialize model
    model = MultitaskTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        num_ner_tags=num_ner_tags
    )
    
    # Create sample inputs
    src = torch.randint(0, vocab_size, (batch_size, seq_length))
    padding_mask = create_padding_mask(src)
    
    # Test classification task
    classification_outputs = model(src, task='classification', src_padding_mask=padding_mask)
    print(f"Classification logits shape: {classification_outputs['logits'].shape}")
    
    # Test NER task
    ner_outputs = model(src, task='ner', src_padding_mask=padding_mask)
    print(f"NER logits shape: {ner_outputs['logits'].shape}")
    
    # Test loss calculation
    criterion = MultitaskLoss()
    
    # Sample labels (ensuring valid ranges)
    classification_labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create NER labels with valid values (0 to num_ner_tags-1) and some padding (-100)
    ner_labels = torch.randint(0, num_ner_tags, (batch_size, seq_length))
    # Add some padding tokens
    ner_labels[:, -2:] = -100  # Set last two tokens as padding
    
    # Calculate losses
    classification_loss = criterion(classification_outputs, classification_labels, 'classification')
    ner_loss = criterion(ner_outputs, ner_labels, 'ner')
    
    print(f"Classification loss: {classification_loss.item()}")
    print(f"NER loss: {ner_loss.item()}")
    
    # Print sample predictions
    print("\nSample predictions:")
    print("Classification predictions:", torch.argmax(classification_outputs['logits'], dim=1))
    print("Classification labels:", classification_labels)
    print("NER predictions (first sequence):", torch.argmax(ner_outputs['logits'][0], dim=1))
    print("NER labels (first sequence):", ner_labels[0])

if __name__ == "__main__":
    test_model()

    