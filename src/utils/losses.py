import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(image_embeddings, text_embeddings, temperature):
    """
    Compute the contrastive loss (InfoNCE/CLIP loss)
    """
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    logits = torch.matmul(text_embeddings, image_embeddings.T) / temperature
    
    # Labels are the diagonal elements
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size).to(logits.device)
    
    # Symmetric cross entropy loss
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2

def compute_retrieval_metrics(image_embeddings, text_embeddings):
    """
    Compute retrieval metrics (R@1, R@5, R@10)
    """
    with torch.no_grad():
        # Compute similarity matrix
        similarities = torch.matmul(text_embeddings, image_embeddings.T)
        
        batch_size = similarities.shape[0]
        labels = torch.arange(batch_size).to(similarities.device)
        
        # Get rankings
        _, indices = similarities.topk(10, dim=1)
        
        # Compute recalls
        r1 = (indices[:, 0] == labels).float().mean()
        r5 = (indices[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean()
        r10 = (indices[:, :10] == labels.unsqueeze(1)).any(dim=1).float().mean()
        
        return r1.item(), r5.item(), r10.item()