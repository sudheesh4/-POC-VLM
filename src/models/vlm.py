import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import timm

class VisionEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True, freeze=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.model.num_features
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", pretrained=True, freeze=True):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)
        
        self.feature_dim = self.model.config.hidden_size
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation as the text embedding
        return outputs.last_hidden_state[:, 0, :]

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim=512, hidden_dim=1024, num_layers=2):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class VLM(nn.Module):
    def __init__(self, vision_encoder, text_encoder, projector_dim=512, temperature=0.1):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        self.image_projector = Projector(vision_encoder.feature_dim, projector_dim)
        self.text_projector = Projector(text_encoder.feature_dim, projector_dim)
        
    def forward(self, images, input_ids, attention_mask):
        # Extract features => text and image both
        with torch.no_grad():
            image_features = self.vision_encoder(images)
            text_features = self.text_encoder(input_ids, attention_mask)
        
        # Project to shared space => shared feature_space embeddings
        image_embeddings = self.image_projector(image_features)
        text_embeddings = self.text_projector(text_features)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        return image_embeddings, text_embeddings, self.temperature

def create_vlm_model(
    vision_model_name="vit_base_patch16_224",
    text_model_name="bert-base-uncased", 
    projector_dim=512,
    freeze_encoders=True
):
    vision_encoder = VisionEncoder(vision_model_name, freeze=freeze_encoders)
    text_encoder = TextEncoder(text_model_name, freeze=freeze_encoders)
    
    model = VLM(vision_encoder, text_encoder, projector_dim=projector_dim)
    return model