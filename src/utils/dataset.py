import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import random

class Flickr30kDataset(Dataset):
    def __init__(self, split="train", image_size=224, text_model_name="bert-base-uncased", max_length=32):
        # Load the entire dataset
        self.full_dataset = load_dataset("nlphuji/flickr30k")
        
        # Check if the requested split exists
        if split not in self.full_dataset:
            available_splits = list(self.full_dataset.keys())
            raise ValueError(f"Unknown split '{split}'. Should be one of {available_splits}.")
        
        self.dataset = self.full_dataset[split]
        self.image_size = image_size
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_length = max_length
        
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # For validation, use deterministic transforms
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.is_train = (split == "train")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Apply appropriate transform
        if self.is_train:
            image = self.image_transform(image)
        else:
            image = self.val_transform(image)
        
        # Randomly select one of the 5 captions during training
        if self.is_train:
            caption = random.choice(item['caption'])
        else:
            # For validation, use the first caption for consistency
            caption = item['caption'][0]
        
        # Tokenize text
        text = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            'image': image,
            'input_ids': text['input_ids'].squeeze(),
            'attention_mask': text['attention_mask'].squeeze(),
            'caption': caption,
            'image_id': idx
        }

def get_dataloaders(batch_size=64, image_size=224, num_workers=2):
    """Create train and validation dataloaders"""
    full_dataset = load_dataset("nlphuji/flickr30k")
    test_dataset = full_dataset['test']
    
    # Split the test dataset into train/val manually
    total_size = len(test_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Create custom dataset class that handles the split
    class SplitFlickr30kDataset(Dataset):
        def __init__(self, base_dataset, indices, is_train=True, image_size=224, text_model_name="bert-base-uncased", max_length=32):
            self.base_dataset = base_dataset
            self.indices = indices
            self.image_size = image_size
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.max_length = max_length
            self.is_train = is_train
            
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.val_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            item = self.base_dataset[actual_idx]
            image = item['image']
            
            if self.is_train:
                image = self.image_transform(image)
            else:
                image = self.val_transform(image)
            
            if self.is_train:
                caption = random.choice(item['caption'])
            else:
                caption = item['caption'][0]
            
            text = self.tokenizer(
                caption, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            
            return {
                'image': image,
                'input_ids': text['input_ids'].squeeze(),
                'attention_mask': text['attention_mask'].squeeze(),
                'caption': caption,
                'image_id': actual_idx
            }
    
    # Create indices for train/val split
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = SplitFlickr30kDataset(test_dataset, train_indices, is_train=True, image_size=image_size)
    val_dataset = SplitFlickr30kDataset(test_dataset, val_indices, is_train=False, image_size=image_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader
