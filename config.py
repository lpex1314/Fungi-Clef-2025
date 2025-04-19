# config.py
'''
This module contains the configuration settings for the model training and evaluation.
It includes parameters such as batch size, learning rate, number of epochs,
device settings, model name, number of classes, image size, and top K predictions.
The settings are encapsulated in a Config class, which can be easily modified
for different training runs.
'''
import torch
from typing import List
# Configuration class
class Config:
    batch_size = 32
    learning_rate = 2e-5
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "hf-hub:imageomics/bioclip"
    num_classes = 0
    image_size = 224
    top_k = 10

config = Config()
