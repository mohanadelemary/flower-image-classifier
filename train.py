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



def data_transform_load(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'


    #  Define your transforms for the training and validation sets
    train_transform = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])


    valid_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)


    # Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    return trainloader, validloader



def build_nn(model_type, hid_units):
    """
    Builds a neural network model using the specified architecture and number of hidden units.
    
    Args:
        model_type (str): The name of the model architecture (e.g., 'vgg13', 'resnet18', 'densenet121').
        hid_units (int): Number of hidden units in the classifier.
    
    Returns:
        model: A PyTorch model with the specified architecture and custom classifier.
    """
    # Dynamically get the model from torchvision.models using the model_type
    try:
        model = getattr(models, model_type)(pretrained=True)
    except AttributeError:
        raise ValueError(f"Model architecture '{model_type}' is not available in torchvision.models.")
    
    # Freeze all feature parameters
    for param in model.parameters():
        param.requires_grad = False

    # Customize the classifier layer for VGG or DenseNet
    if hasattr(model, 'classifier'):
        input_features = model.classifier[0].in_features  # Fetch input features dynamically
        model.classifier = Classifier_Layer(input_features, 102, [hid_units])
    
    # Customize the fully connected (fc) layer for ResNet or similar architectures
    elif hasattr(model, 'fc'):
        input_features = model.fc.in_features  # Fetch input features dynamically
        model.fc = Classifier_Layer(input_features, 102, [hid_units])
    
    else:
        raise ValueError(f"Model architecture '{model_type}' does not have a recognized classifier or fc layer.")
    
    return model


def train_model(model, trainloader, validloader, num_epochs,learn_rate):

    best_accuracy = 0  # Initialize to track the best accuracy
    best_model_path = "best_model_classifier.pth"  # Path to save the best model

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    epochs = num_epochs
    steps = 0

    train_losses, valid_losses = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for e in range(epochs):
        running_loss = 0
        print(f"Starting epoch {e+1}/{epochs}")

        for images, labels in trainloader:
            steps += 1

            # Move data to the device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            log_ps = model(images)
            loss = criterion(log_ps, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Perform validation every 4 steps
            if steps % 4 == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()  # Set the model to evaluation mode

                with torch.no_grad():
                    for val_images, val_labels in validloader:
                        val_images, val_labels = val_images.to(device), val_labels.to(device)

                        # Forward pass
                        log_ps = model(val_images)
                        batch_loss = criterion(log_ps, val_labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == val_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                 # Log stats
                train_loss_avg = running_loss / len(trainloader)
                valid_loss_avg = valid_loss / len(validloader)
                valid_accuracy = accuracy / len(validloader)

                train_losses.append(train_loss_avg)
                valid_losses.append(valid_loss_avg)

                print(f"Step {steps}.. "
                      f"Train Loss: {train_loss_avg:.3f}.. "
                      f"Validation Loss: {valid_loss_avg:.3f}.. "
                      f"Validation Accuracy: {valid_accuracy*100:.2f}%")

                # Save the best model
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    best_model = torch.save(model.classifier.state_dict(), best_model_path)
                    print(f"Saved best model with accuracy: {best_accuracy * 100:.2f}%")
                else:
                    print("Model shows no improvement. Model not saved.")

            model.train()  # Set the model back to training mode

    # Print summary after the epoch
    print(f"Epoch {e+1}/{epochs} completed.")

    
    return best_model


def save_checkpoint(best_model, save_d):
    
    
    # Save the checkpoint with fixed hidden layers extraction
    model.to('cpu')  # Ensure the model is on CPU for saving

    # Create the checkpoint
    checkpoint = {
        'input_size': 25088,  # Input size of the model (e.g., for VGG16)
        'output_size': 102,   # Number of output classes
        'hidden_layers': [layer.out_features for layer in model.classifier.hidden_layers],
        'drop_p': 0.2,  # Dropout probability (if applicable)
        'state_dict': model.state_dict(),  # Model's state_dict
        'class_to_idx': model.class_to_idx  # Class-to-index mapping
    }

    checkpoint_file_name = 'checkpoint.pth'
    checkpoint_path = save_d + checkpoint_file_name

    # Save the model
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {filepath}")

    return checkpoint_path



def main():
    
    
     # Get arguments for training mode
    args = get_input_args(mode='train')

    # Display the parsed arguments
    print(f"Training with the following parameters:\n"
          f"Data Directory: {args.data_dir}\n"
          f"Save Directory: {args.save_dir}\n"
          f"Architecture: {args.arch}\n"
          f"Learning Rate: {args.learning_rate}\n"
          f"Hidden Units: {args.hidden_units}\n"
          f"Epochs: {args.epochs}\n"
          f"GPU Enabled: {args.gpu}")
    
    print("Loading Data ...")
    trainloader, validloader = data_transform_load(data_dir = args.data_dir)
    
    print("Building Model ...")

    model = build_nn(model_type = args.arch, hid_units = args.hidden_units)

    print("Initiating model training ...")
    
    best_model = train_model(model, trainloader, validloader, num_epochs = args.epochs,learn_rate = args.learning_rate)

    print("Saving Model as .pth file ...")
    checkpoint_path = save_checkpoint(best_model, save_d = args.save_dir)

    print(f"Model Checkpoint has been saved at {checkpoint_path}")

if __name__ == "__main__":
    
    main()




