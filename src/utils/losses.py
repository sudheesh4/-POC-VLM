import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(image_embeddings, text_embeddings, temperature):
    """
    Compute the contrastive loss (InfoNCE/CLIP loss)
    """
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1) #shape : (N_img=Batch, Embed_dim)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)  #shape : (N_text=Batch, Embed_dim)
    
    # Compute similarity matrix
    logits = torch.matmul(text_embeddings, image_embeddings.T) / temperature  ##shape : (N_text, N_img) 
    
    # Labels are the diagonal elements
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size).to(logits.device)  # N img-text pairs #shape (N=Batch,)
    
    # Symmetric cross entropy loss
    
    loss_i = F.cross_entropy(logits, labels)   # for each text, find correct image
    # Text-to-image loss : n_th text should be linked to n_th image only (highest similarity for n_th pair)
    # Single scalar loss value averaged over all batch
    
    loss_t = F.cross_entropy(logits.T, labels) # for each image, find correct text 
    # Image-to-text loss : n_th image should be linked to n_th text only (highest similarity for n_th pair)
    # Single scalar loss value averaged over all batch
    
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
