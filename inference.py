import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

def load_vlm_for_inference(checkpoint_path, device=None,load_new=False):
    """
    Load a trained VLM model from checkpoint for inference
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config 
    config = checkpoint.get('config', {
        'vision_model': 'vit_base_patch16_224',
        'text_model': 'bert-base-uncased',
        'projector_dim': 512
    })
    
    # model architecture
    from src.models.vlm import create_vlm_model  # Import model class
    model = create_vlm_model(
        vision_model_name=config['vision_model'],
        text_model_name=config['text_model'],
        projector_dim=config['projector_dim'],
        freeze_encoders=True
    )
    

    if not load_new:
    # Load trained weights
      model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    
    print(f"Model loaded successfully on {device}")
    print(f"Vision encoder: {config['vision_model']}")
    print(f"Text encoder: {config['text_model']}")
    print(f"Projector dimension: {config['projector_dim']}")
    
    return model, config, tokenizer, device

def create_image_transform(image_size=224):
    """
    Create image transformation pipeline (same as validation during training)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def image_similarity_inference(checkpoint_path, image_path, prompts, 
                             temperature=None, return_embeddings=False,load_new=False):
    """
    Main function: Load model and compute similarity between image and prompts
    """
    # Load model and components
    model, config, tokenizer, device = load_vlm_for_inference(checkpoint_path,load_new=load_new)
    transform = create_image_transform(config.get('image_size', 224))
    
    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Use learned temperature if not specified
    if temperature is None:
        temperature = model.temperature.item()
        print(f"Using learned temperature: {temperature:.4f}")
    
    results = {
        'image_path': image_path,
        'prompts': prompts,
        'temperature': temperature,
        'similarities': [],
        'probabilities': [],
        'ranking': []
    }
    
    # Encode image once
    with torch.no_grad():
        image_features = model.vision_encoder(image_tensor)
        image_embedding = model.image_projector(image_features)
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
    
    # Encode each prompt and compute similarity
    prompt_embeddings = []
    
    for prompt in prompts:
        # Tokenize text
        inputs = tokenizer(
            prompt, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=32,
            return_token_type_ids=False 
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode text
        with torch.no_grad():
            text_features = model.text_encoder(**inputs)
            text_embedding = model.text_projector(text_features)
            text_embedding = F.normalize(text_embedding, p=2, dim=1)
        
        prompt_embeddings.append(text_embedding)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(image_embedding, text_embedding)
        results['similarities'].append(similarity.item())
    
    # Convert to probabilities using softmax + temperature
    similarities_tensor = torch.tensor(results['similarities'])
    probabilities = F.softmax(similarities_tensor / temperature, dim=0)
    results['probabilities'] = probabilities.tolist()
    
    # Create ranked list
    ranked_indices = torch.argsort(similarities_tensor, descending=True)
    results['ranking'] = [(prompts[i], results['probabilities'][i]) 
                         for i in ranked_indices]
    
    if return_embeddings:
        results['image_embedding'] = image_embedding.cpu()
        results['prompt_embeddings'] = torch.cat(prompt_embeddings).cpu()
    
    return results

def print_similarity_results(results, top_k=None,load_new=False):
    """
    Pretty print the similarity results
    """
    print(f"\n{'='*60}")
    print(f"Model : {'Un-Trained' if load_new else 'Trained'}")
    print(f"IMAGE: {results['image_path']}")
    print(f"TEMPERATURE: {results['temperature']:.4f}")
    print(f"{'='*60}")
    print(f"{'RANK':<4} {'PROMPT':<40} {'SIMILARITY':<12} {'PROBABILITY':<12}")
    print(f"{'-'*60}")
    
    ranking = results['ranking']
    if top_k is not None:
        ranking = ranking[:top_k]
    
    for rank, (prompt, prob) in enumerate(ranking, 1):
        # Find original similarity score
        prompt_idx = results['prompts'].index(prompt)
        similarity = results['similarities'][prompt_idx]
        
        print(f"{rank:<4} {prompt:<40} {similarity:<12.4f} {prob:<12.4f}")


checkpoint_path = os.path.join("./checkpoints", 'best_model.pth')
image_path = "./dog.jpg"
prompts = [
    "a photo of a dog",
    "a photo of a cat", 
    "a photo of a car",
    "a photo of a tree",
    "a photo of a building",
    "a photo of a flower"
]




results_trained = image_similarity_inference(checkpoint_path, image_path, prompts,load_new=False)
results_untrained = image_similarity_inference(checkpoint_path, image_path, prompts,load_new=True)

print_similarity_results(results_untrained,load_new=True)
print_similarity_results(results_trained,load_new=False)
