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
