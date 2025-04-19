'''
The following code is a script for training and evaluating a Prototypical Network model on the FungiTastic dataset. 
It includes functions for computing class prototypes, training and evaluating the model, and saving predictions. 
The script uses PyTorch and OpenCLIP for model creation and image preprocessing.
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import open_clip
import matplotlib.pyplot as plt
from typing import List
import warnings

from models import FungiTastic, PrototypicalLoss, FungiEmbedder
from config import config

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()


def compute_prototypes(model, dataloader, device, num_classes):
    """
    Compute class prototypes by averaging embeddings from support samples
    for each class in the dataset.
    Args:
        model: FungiEmbedder
        dataloader: training dataloader
        device: cuda/cpu
        num_classes: number of classes in the dataset
    Returns:
        prototypes: tensor [num_classes, embedding_dim]
    """
    model.eval()
    all_embeddings = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Prototypes"):
            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            for emb, label in zip(embeddings, labels):
                all_embeddings[label.item()].append(emb.cpu())
            # delete to free up memory
            del images, labels, embeddings
            torch.cuda.empty_cache()

    prototypes = []
    for class_embs in all_embeddings:
        if len(class_embs) == 0:
            prototypes.append(torch.zeros_like(embeddings[0]))
        else:
            prototypes.append(torch.stack(class_embs).mean(dim=0))

    return torch.stack(prototypes).to(device)


def train_epoch_protonet(model, dataloader, criterion, optimizer, device, prototypes):
    """
    Train one epoch using Prototypical Network logic.
    Args:
        model: FungiEmbedder
        dataloader: training dataloader
        criterion: PrototypicalLoss
        optimizer: optimizer
        device: cuda/cpu
        prototypes: tensor [num_classes, embedding_dim]
    Returns:
        epoch_loss: average loss over the epoch
        accuracy: accuracy score
        f1: F1 score
    """
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss, logits = criterion(embeddings, labels, prototypes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        epoch_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss / len(dataloader), accuracy, f1


def evaluate_epoch_protonet(model, dataloader, criterion, device, prototypes):
    '''
    Evaluate one epoch using Prototypical Network logic.
    Args:
        model: FungiEmbedder
        dataloader: validation dataloader
        criterion: PrototypicalLoss
        device: cuda/cpu
        prototypes: tensor [num_classes, embedding_dim]
    Returns:
        epoch_loss: average loss over the epoch
        accuracy: accuracy score
        f1: F1 score
    '''
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            loss, logits = criterion(embeddings, labels, prototypes)

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss / len(dataloader), accuracy, f1

def test_collate_fn(batch):
    """
    Custom collate function for test dataloader,
    allows category_id to be None.
    Args:
        batch: list of tuples (image, label, file_path, observation_id)
    Returns:
        images: tensor [B, C, H, W]
        labels: list of None
        file_paths: list of file paths
        observation_ids: list of observation IDs
    """
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]  # will be all None
    file_paths = [item[2] for item in batch]
    observation_ids = [item[3] for item in batch]
    return images, labels, file_paths, observation_ids


# ## Mean Pooling
def evaluate_protonet_grouped(model, dataloader, prototypes, device, dataset, k=10, save_path="test_predictions.csv"):
    """
    Evaluate using prototype similarity and group predictions by observation ID.
    Aggregates predictions per observation using mean pooling.
    Args:
        model: FungiEmbedder
        dataloader: test dataloader
        prototypes: tensor [num_classes, embedding_dim]
        device: cuda/cpu
        dataset: FungiTastic (for ID mapping)
        k: number of top predictions
        save_path: CSV save path
    Returns:
        df: DataFrame with predictions
    """
    model.eval()
    observation_logits = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, _, _, observation_ids = batch
            images = images.to(device)
            embeddings = model(images)
            dists = torch.cdist(embeddings, prototypes, p=2)  # [B, C]
            probs = -dists  # higher is better

            probs = probs.cpu().numpy()
            for i, obs_id in enumerate(observation_ids):
                if obs_id not in observation_logits:
                    observation_logits[obs_id] = []
                observation_logits[obs_id].append(probs[i])

    # Aggregate per observation
    results = []
    for obs_id, logits_list in observation_logits.items():
        avg_logits = np.mean(logits_list, axis=0)
        topk = np.argsort(avg_logits)[-k:][::-1]
        predictions = ' '.join(str(c) for c in topk)
        results.append({'ObservationId': obs_id, 'predictions': predictions})

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Saved grouped prediction results to {save_path}")
    return df


if __name__ == "__main__":
    '''
    Main function to train and evaluate the Prototypical Network model.
    It loads the FungiTastic dataset, initializes the model, and trains it.
    After training, it evaluates the model on the test set and saves predictions.
    '''
    data_root = "data/fungi-clef-2025"

    # Load BioCLIP model and preprocessing function
    model, _, preprocess = open_clip.create_model_and_transforms(config.model_name)
    embedder = FungiEmbedder(model).to(config.device)

    # Load datasets
    train_dataset = FungiTastic(root=data_root, split='train', transform=preprocess)
    val_dataset = FungiTastic(root=data_root, split='val', transform=preprocess)
    test_dataset = FungiTastic(root=data_root, split='test', transform=preprocess)

    config.num_classes = train_dataset.n_classes

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize loss function and optimizer
    criterion = PrototypicalLoss()
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)


    # Compute class prototypes from training data
    prototypes = compute_prototypes(embedder, train_loader, config.device, config.num_classes)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []

    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    checkpoint_path = os.path.join(data_root, "best_fungiembedder.pt")

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        # Train one epoch
        train_loss, train_acc, train_f1 = train_epoch_protonet(embedder, train_loader, criterion, optimizer, config.device, prototypes)
        scheduler.step()
        # Evaluate on validation set
        val_loss, val_acc, val_f1 = evaluate_epoch_protonet(embedder, val_loader, criterion, config.device, prototypes)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # Checkpoint saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': embedder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✅ New best model saved! Validation Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break
        
    # save best model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': embedder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "best_fungiembedder.pt")

    # Load the best model
    checkpoint_path = 'best_fungiembedder.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        embedder.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {checkpoint_path}")

    # Evaluate on test set

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=test_collate_fn
    )

    evaluate_protonet_grouped(
        model=embedder,
        dataloader=test_loader,
        prototypes=prototypes,
        device=config.device,
        dataset=test_dataset,
        k=config.top_k,
        save_path="results/test_predictions_protoNet_grouped.csv"
)
