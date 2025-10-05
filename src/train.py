import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import sys


from utils.dataset import get_dataloaders
from utils.losses import contrastive_loss, compute_retrieval_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from models.vlm import create_vlm_model

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup model
        self.model = create_vlm_model(
            vision_model_name=config['vision_model'],
            text_model_name=config['text_model'],
            projector_dim=config['projector_dim'],
            freeze_encoders=True
        ).to(self.device)
        print(">>>>>MODEL CREATED")
        # Setup data
        self.train_loader, self.val_loader = get_dataloaders(
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            num_workers=2
        )
        
        # Setup optimizer (only train projectors and temperature)
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'projector' in name or 'temperature' in name:
                trainable_params.append(param)
        
        self.optimizer = AdamW(trainable_params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            image_emb, text_emb, temperature = self.model(images, input_ids, attention_mask)
            
            # Compute loss
            loss = contrastive_loss(image_emb, text_emb, temperature)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_r1, total_r5, total_r10 = 0, 0, 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                image_emb, text_emb, temperature = self.model(images, input_ids, attention_mask)
                
                loss = contrastive_loss(image_emb, text_emb, temperature)
                total_loss += loss.item()
                
                # Compute retrieval metrics
                r1, r5, r10 = compute_retrieval_metrics(image_emb, text_emb)
                total_r1 += r1
                total_r5 += r5
                total_r10 += r10
        
        avg_loss = total_loss / num_batches
        avg_r1 = total_r1 / num_batches
        avg_r5 = total_r5 / num_batches
        avg_r10 = total_r10 / num_batches
        
        metrics = {
            'val_loss': avg_loss,
            'r1': avg_r1,
            'r5': avg_r5,
            'r10': avg_r10
        }
        
        return metrics
    
    def train(self, resume=False):
        start_epoch = 0
        
        # Resume from checkpoint if requested
        if resume:
            checkpoint_path = get_latest_checkpoint(self.config['checkpoint_dir'])
            if checkpoint_path:
                checkpoint = load_checkpoint(self.model, self.optimizer, self.scheduler, checkpoint_path)
                if checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    self.best_loss = checkpoint['metrics']['val_loss']
        
        print(f"Starting training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.config['epochs']):
            self.current_epoch = epoch
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}')
            
            # Training
            start_time = time.time()
            train_loss = self.train_epoch()
            epoch_time = time.time() - start_time
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Print results
            print(f'Epoch {epoch+1} - Time: {epoch_time:.2f}s')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_metrics["val_loss"]:.4f}')
            print(f'  R@1: {val_metrics["r1"]:.4f}, R@5: {val_metrics["r5"]:.4f}, R@10: {val_metrics["r10"]:.4f}')
            print(f'  LR: {self.scheduler.get_last_lr()[0]:.2e}')
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'], 
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, 
                    epoch, val_metrics['val_loss'], val_metrics, 
                    checkpoint_path
                )
            
            # Save best model
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics['val_loss'], val_metrics,
                    best_path
                )
                print(f'  New best model saved! Loss: {self.best_loss:.4f}')

def main():
    config = {
        # Model
        'vision_model': 'vit_base_patch16_224',
        'text_model': 'bert-base-uncased',
        'projector_dim': 512,
        
        # Training
        'batch_size': 64,
        'image_size': 224,
        'epochs': 20,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        
        # Checkpointing
        'checkpoint_dir': './vlm-project/checkpoints',
        'checkpoint_interval': 1,  # Save every epoch
        
        # Resume
        'resume': True
    }

    trainer = Trainer(config)
    print(">>>>Starting Training")
    trainer.train(resume=config['resume'])

if __name__ == '__main__':
    main()