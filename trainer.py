import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
from multitask_learning import (
    MultitaskTransformer,
    MultitaskLoss,
)

def pad_or_truncate(seq, max_len, pad_value=-100):
    return seq[:max_len] + [pad_value] * max(0, max_len - len(seq))


class MultitaskDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        classification_labels: List[int],
        ner_labels: List[List[int]],
        tokenizer,  # Would be actually implemented with a real tokenizer
        max_length: int = 512
    ):
        """
        Dataset for multi-task learning.
        Args:
            texts: List of input texts
            classification_labels: List of classification labels
            ner_labels: List of NER label sequences
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.classification_labels = classification_labels
        self.ner_labels = ner_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Simulated tokenization for demonstration
        # In reality, would use actual tokenizer
        input_ids = torch.randint(0, 1000, (self.max_length,))
        attention_mask = torch.ones(self.max_length)
        attention_mask[-2:] = 0  # Simulate some padding
        
        padded_ner_labels = pad_or_truncate(self.ner_labels[idx], self.max_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'classification_label': torch.tensor(self.classification_labels[idx]),
            'ner_labels': torch.tensor(padded_ner_labels)
        }

class MultitaskTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Trainer for multi-task learning model.
        Args:
            model: MultitaskTransformer model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = MultitaskLoss()
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        task_losses = {'classification': 0, 'ner': 0}
        task_counts = {'classification': 0, 'ner': 0}
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Randomly choose task for this batch
            task = random.choice(['classification', 'ner'])
            
            # Forward pass
            outputs = self.model(
                src=batch['input_ids'],
                task=task,
                src_padding_mask=batch['attention_mask'] == 0
            )
            
            print(f"Batch Size: {batch['input_ids'].shape[0]}")
            print(f"Output Shape: {outputs['logits'].shape}")
            print(f"NER Labels Shape: {batch['ner_labels'].shape}")  # Ensure they match
            
            # Calculate loss
            if task == 'classification':
                loss = self.criterion(outputs, batch['classification_label'], task)
            else:  # NER
                loss = self.criterion(outputs, batch['ner_labels'], task)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            task_losses[task] += loss.item()
            task_counts[task] += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'task': task
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_task_losses = {
            task: loss / count 
            for task, (loss, count) in zip(
                task_losses.keys(),
                zip(task_losses.values(), task_counts.values())
            )
        }
        
        return {
            'loss': avg_loss,
            **avg_task_losses
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        task_losses = {'classification': 0, 'ner': 0}
        task_counts = {'classification': 0, 'ner': 0}
        
        classification_preds = []
        classification_labels = []
        ner_preds = []
        ner_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Evaluate classification
                outputs = self.model(
                    src=batch['input_ids'],
                    task='classification',
                    src_padding_mask=batch['attention_mask'] == 0
                )
                loss = self.criterion(outputs, batch['classification_label'], 'classification')
                task_losses['classification'] += loss.item()
                task_counts['classification'] += 1
                
                classification_preds.extend(
                    torch.argmax(outputs['logits'], dim=1).cpu().numpy()
                )
                classification_labels.extend(
                    batch['classification_label'].cpu().numpy()
                )
                
                # Evaluate NER
                outputs = self.model(
                    src=batch['input_ids'],
                    task='ner',
                    src_padding_mask=batch['attention_mask'] == 0
                )
                loss = self.criterion(outputs, batch['ner_labels'], 'ner')
                task_losses['ner'] += loss.item()
                task_counts['ner'] += 1
                
                ner_preds.extend(
                    torch.argmax(outputs['logits'], dim=2).cpu().numpy()
                )
                ner_labels.extend(
                    batch['ner_labels'].cpu().numpy()
                )
        
        # Calculate metrics
        classification_accuracy = np.mean(
            np.array(classification_preds) == np.array(classification_labels)
        )
        
        # Calculate NER accuracy only on non-padding tokens
        mask = np.array(ner_labels) != -100
        ner_accuracy = np.mean(
            np.array(ner_preds)[mask] == np.array(ner_labels)[mask]
        )
        
        return {
            'classification_loss': task_losses['classification'] / task_counts['classification'],
            'ner_loss': task_losses['ner'] / task_counts['ner'],
            'classification_accuracy': classification_accuracy,
            'ner_accuracy': ner_accuracy
        }

def train_model(
    model: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    warmup_steps: int = 1000,
    weight_decay: float = 0.01
) -> Dict[str, List[float]]:
    """
    Train the multi-task model.
    Args:
        model: MultitaskTransformer model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay coefficient
    Returns:
        Dictionary of training history
    """
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )
    
    # Initialize trainer
    trainer = MultitaskTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_classification_loss': [],
        'val_ner_loss': [],
        'val_classification_accuracy': [],
        'val_ner_accuracy': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch()
        history['train_loss'].append(train_metrics['loss'])
        
        # Evaluate
        val_metrics = trainer.evaluate()
        history['val_classification_loss'].append(val_metrics['classification_loss'])
        history['val_ner_loss'].append(val_metrics['ner_loss'])
        history['val_classification_accuracy'].append(val_metrics['classification_accuracy'])
        history['val_ner_accuracy'].append(val_metrics['ner_accuracy'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Classification Loss: {val_metrics['classification_loss']:.4f}")
        print(f"Val NER Loss: {val_metrics['ner_loss']:.4f}")
        print(f"Val Classification Accuracy: {val_metrics['classification_accuracy']:.4f}")
        print(f"Val NER Accuracy: {val_metrics['ner_accuracy']:.4f}")
    
    return history

# Example usage
def test_training():
    # Create dummy data
    texts = ["Sample text " + str(i) for i in range(100)]
    classification_labels = [random.randint(0, 2) for _ in range(100)]
    ner_labels = [
        [random.randint(0, 4) if j < 8 else -100 for j in range(10)]
        for _ in range(100)
    ]
    
    # Create datasets
    train_dataset = MultitaskDataset(
        texts[:80],
        classification_labels[:80],
        ner_labels[:80],
        tokenizer=None  # Would use actual tokenizer in practice
    )
    
    val_dataset = MultitaskDataset(
        texts[80:],
        classification_labels[80:],
        ner_labels[80:],
        tokenizer=None
    )
    
    # Initialize model
    model = MultitaskTransformer(
        vocab_size=1000,
        num_classes=3,
        num_ner_tags=5
    )
    
    # Train model
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=3  # Reduced for testing
    )
    
    return history

if __name__ == "__main__":
    history = test_training()

    