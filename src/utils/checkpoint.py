import torch
import os
import json
from datetime import datetime

def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, filepath):
    """
    Save training checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, scheduler, filepath):
    """
    Load training checkpoint
    """
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return None
    
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    print(f"Resuming from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    
    return checkpoint

def get_latest_checkpoint(checkpoint_dir):
    """
    Get the most recent checkpoint file
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])