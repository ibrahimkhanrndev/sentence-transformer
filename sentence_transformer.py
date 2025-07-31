import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 5000
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: embedding dimension
            max_len: maximum sequence length
        """
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: input tensor of shape [seq_len, batch_size, embedding_dim]

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0)]

class SentenceTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        """
        Initialize the SentenceTransformer model.

        Args:
            vocab_size: size of vocabulary
            d_model: embedding dimension
            nhead: number of attention heads
            num_encoder_layers: number of transformer encoder layers
            dim_feedforward: dimension of feedforward network
            dropout: dropout probability
            max_seq_length: maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            num_encoder_layers
        )
        
        # Global average pooling layer to get fixed-length sentence embedding
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(
        self,
        src: Tensor,
        src_mask: Tensor = None,
        src_padding_mask: Tensor = None
    ) -> Tensor:
        """
        Forward pass of the model.

        Args:
            src: input tensor of shape [batch_size, seq_len]
            src_mask: mask for self-attention
            src_padding_mask: mask for padding tokens
        
        Returns:
            Tensor of shape [batch_size, d_model] representing sentence embeddings
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
        
        # Global average pooling across sequence length
        # Transpose to move sequence length to last dimension
        pooled = self.pooling(encoded.transpose(1, 2))
        
        # Squeeze the pooling dimension and return
        return pooled.squeeze(-1)

def create_padding_mask(
    src: Tensor, 
    pad_idx: int = 0
) -> Tensor:
    """
    Create padding mask for transformer input.
    
    Args:
        src: input tensor of shape [batch_size, seq_len]
        pad_idx: index used for padding
    
    Returns:
        Boolean mask tensor where True indicates padding positions
    """
    return (src == pad_idx)

# Example usage
def test_model():
    # Create a small vocabulary for testing
    vocab_size = 1000
    batch_size = 2
    seq_length = 10
    
    # Initialize model
    model = SentenceTransformer(vocab_size=vocab_size)
    
    # Create sample input
    src = torch.randint(0, vocab_size, (batch_size, seq_length))
    padding_mask = create_padding_mask(src)
    
    # Get sentence embeddings
    with torch.no_grad():
        embeddings = model(src, src_padding_mask=padding_mask)
    
    print(f"Input shape: {src.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    return embeddings

if __name__ == "__main__":
    test_model()

    