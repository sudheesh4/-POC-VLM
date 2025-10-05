# Vision-Language Model (VLM) Project

This (POC-)project trains a VLM using frozen pre-trained vision and text encoders, with learnable projector networks that map both modalities into a shared semantic space. 
The model enables zero-shot image classification and cross-modal retrieval by computing similarity between image and text embeddings.

## Project Structure
```
vlm-project/
├── src/
│   ├── models/
│   │   └── vlm.py              # Model architecture
│   ├── utils/
│   │   ├── dataset.py          # Data loading and preprocessing
│   │   ├── losses.py           # Contrastive loss implementation
│   │   └── checkpoint.py       # Checkpoint saving/loading
│   └── train.py                # Main training script
├── checkpoints/                # Saved models and training states
├── data/                       # Dataset storage
└── inference.py               # Inference and evaluation utilities
```



### Features
- **Frozen Encoders**: Uses pre-trained vision (ViT) and text (BERT) encoders
- **Contrastive Learning**: Implements InfoNCE loss for alignment
- **Checkpointing**: Automatic saving/resuming of training progress
- **Efficient Training**: Only projector networks are trained
- **Zero-shot Inference**: Classify images using natural language prompts
- **Comprehensive Evaluation**: Retrieval metrics (R@1, R@5, R@10)

## Model Architecture

```
Input Image → Frozen Vision Encoder → Image Projector → Shared Embedding Space
                                                       ↗
Input Text  → Frozen Text Encoder   → Text Projector  →
```

### Components:
- **Vision Encoder**: Pre-trained ViT (frozen)
- **Text Encoder**: Pre-trained BERT (frozen) 
- **Projector Networks**: Trainable MLPs that map to shared space
- **Contrastive Loss**: Symmetric InfoNCE loss for alignment between modalities


## Results

After training on Flickr30k for 20 Epochs:

  Train Loss: 0.9366
  Val Loss: 1.1320
  R@1: 0.6621, R@5: 0.9262, R@10: 0.9702

### Zero-shot Image Classification

MODEL : Un-Trained
IMAGE: /dog.jpg
TEMPERATURE: 0.1000

RANK; PROMPT;                                   SIMILARITY;   PROBABILITY 
------------------------------------------------------------
1 ;   a photo of a tree                       ; -0.0644     ; 0.1716      
2 ;   a photo of a car                        ; -0.0657     ; 0.1694      
3 ;   a photo of a flower                     ; -0.0675     ; 0.1663      
4 ;   a photo of a cat                        ; -0.0682     ; 0.1653      
5 ;   a photo of a dog                        ; -0.0690     ; 0.1639      
6 ;   a photo of a building                   ;  -0.0692    ;  0.1636      

MODEL : Trained
IMAGE: /dog.jpg
TEMPERATURE: 0.0395

RANK; PROMPT;                                   SIMILARITY;   PROBABILITY 
------------------------------------------------------------
1 ;    a photo of a dog                       ; 0.7622      ; 0.7112      
2 ;    a photo of a cat                       ;  0.7255     ;  0.2810      
3 ;   a photo of a flower                     ; 0.5711      ; 0.0056      
4 ;   a photo of a car                        ; 0.5237      ; 0.0017      
5 ;   a photo of a tree                       ; 0.4673      ; 0.0004      
6 ;   a photo of a building                   ; 0.3643      ; 0.0000      

