import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time
from data_prep import get_input_args, Classifier_Layer

print(torch.__version__)
print(torch.cuda.is_available()) 
torch.cuda.empty_cache()



def preprocess(image_path):

    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    

    # Open the image
    pil_image = Image.open(image_path)

    # Resize so the shortest side is 256 pixels while maintaining aspect ratio
    pil_image = pil_image.resize((256, int(pil_image.height * 256 / pil_image.width)) 
                                  if pil_image.width < pil_image.height 
                                  else (int(pil_image.width * 256 / pil_image.height), 256))

    # Center crop the image to 224x224
    width, height = pil_image.size
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert the image to a NumPy array
    np_image = np.array(pil_image) / 255.0  # Scale pixel values to [0, 1]

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions to CxHxW
    pred_img = np_image.transpose((2, 0, 1))

    
    return pred_img


def load_checkpoint(checkpoint_path, arch):
    
    checkpoint = torch.load(checkpoint_path)
    
    model = models.arch(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = Classifier_Layer(checkpoint['input_size'],checkpoint['output_size'], checkpoint['hidden_layers'])
    
    model.load_state_dict(checkpoint['state_dict']) 
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model



def predict(pred_img, model, category_names, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse mapping

    # Process the image
    
    image_tensor = torch.from_numpy(pred_img).float().unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        model.eval()
        logits = model(image_tensor)  # Forward pass
        ps = torch.exp(logits)  # Convert logits to probabilities
        top_p, top_class = ps.topk(top_k, dim=1)  # Get top-k predictions
    
    
    # Convert top_class to corresponding folder labels
    top_classes = top_class.squeeze().cpu().numpy()  # Get top class indices
    folder_labels = [idx_to_class[i] for i in top_classes]  # Map indices to folder labels

    # Map folder labels to flower names using cat_to_name
    flower_names = [cat_to_name[label] for label in folder_labels]
    
    # Prepare results
    results = list(zip(flower_names, top_p.squeeze().cpu().numpy()))

    # Display predictions
    print("Predictions:")
    for name, prob in results:
        print(f"{name}: {prob:.3f}")
    
    
    return results


def main():
    
    # Get arguments for prediction mode
    args = get_input_args(mode='predict')
    
    # Example usage of arguments
    print(f"Predicting with the following parameters:\n"
          f"Input Image: {args.input}\n"
          f"Checkpoint: {args.checkpoint}\n"
          f"Top K: {args.top_k}\n"
          f"Category Names: {args.category_names}\n"
          f"GPU Enabled: {args.gpu}")
    
    pred_img = preprocess(image_path=args.input)

    model = load_checkpoint(checkpoint_path = args.checkpoint, arch = args.arch)

    results = predict(pred_img, model, category_names = args.category_names, top_k=args.top_k)


if __name__ == "__main__":
    
    main()

