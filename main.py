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


# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Configuration class
class Config:
    batch_size = 32
    learning_rate = 2e-5
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "hf-hub:imageomics/bioclip"
    num_classes = 0
    image_size = 224
    top_k = 10

config = Config()

class FungiTastic(torch.utils.data.Dataset):
    """
    Dataset class for the FewShot subset of the Danish Fungi dataset (size 300, closed-set).

    This dataset loader supports training, validation, and testing splits, and provides
    convenient access to images, class IDs, and file paths. It also supports optional
    image transformations.
    """

    SPLIT2STR = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

    def __init__(self, root: str, split: str = 'val', transform=None):
        """
        Initializes the FungiTastic dataset.

        Args:
            root (str): The root directory of the dataset.
            split (str, optional): The dataset split to use. Must be one of {'train', 'val', 'test'}.
                Defaults to 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.split = split
        self.transform = transform
        self.df = self._get_df(root, split)
        self.feature_df = self.df.copy()
        self.feature_selection = ['year', 'month', 'habitat', 'country_code', 'iucnRedListCategory', 'substrate', 'coorUncert', 'latitude', 'longitude', 'region', 'district', 'poisonous', 'elevation', 'landcover', 'biogeographicalRegion']

        assert "image_path" in self.df
        if self.split != 'test':
            assert "category_id" in self.df
            self.n_classes = len(self.df['category_id'].unique())
            self.category_id2label = {
                k: v[0] for k, v in self.df.groupby('category_id')['species'].unique().to_dict().items()
            }
            self.label2category_id = {
                v: k for k, v in self.category_id2label.items()
            }
        else:
            # For test set, we need to load category IDs from training set
            train_df = self._get_df(root, 'train')
            self.n_classes = len(train_df['category_id'].unique())
            self.category_id2label = {
                k: v[0] for k, v in train_df.groupby('category_id')['species'].unique().to_dict().items()
            }
            self.label2category_id = {
                v: k for k, v in self.category_id2label.items()
            }

    def add_embeddings(self, embeddings: pd.DataFrame):
        """
        Updates the dataset instance with new embeddings.

        Args:
            embeddings (pd.DataFrame): A DataFrame containing an 'embedding' column.
                                       It must align with `self.df` in terms of indexing.
        """
        assert isinstance(embeddings, pd.DataFrame), "Embeddings must be a pandas DataFrame."
        assert "embedding" in embeddings.columns, "Embeddings DataFrame must have an 'embedding' column."
        assert len(embeddings) == len(self.df), "Embeddings must match dataset length."

        self.df = pd.merge(self.df, embeddings, on="filename", how="inner")

    def get_embeddings_for_class(self, id):
        # return the embeddings for class class_idx
        class_idxs = self.df[self.df['category_id'] == id].index
        return self.df.iloc[class_idxs]['embedding']

    @staticmethod
    def _get_df(data_path: str, split: str) -> pd.DataFrame:
        """
        Loads the dataset metadata as a pandas DataFrame.

        Args:
            data_path (str): The root directory where the dataset is stored.
            split (str): The dataset split to load. Must be one of {'train', 'val', 'test'}.

        Returns:
            pd.DataFrame: A DataFrame containing metadata and file paths for the split.
        """
        df_path = os.path.join(
            data_path,
            "metadata",
            "FungiTastic-FewShot",
            f"FungiTastic-FewShot-{FungiTastic.SPLIT2STR[split]}.csv"
        )
        df = pd.read_csv(df_path)
        df["image_path"] = df.filename.apply(
            lambda x: os.path.join(data_path, "FungiTastic-FewShot", split, '300p', x)
        )
        return df

    def __getitem__(self, idx: int):
        """
        Retrieves a single data sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, category_id, file_path, observation_id, embedding)
        """
        file_path = self.df["image_path"].iloc[idx].replace('FungiTastic-FewShot', 'images/FungiTastic-FewShot')
        
        # Get observationID if available
        observation_id = self.df["observationID"].iloc[idx] if "observationID" in self.df.columns else None

        if self.split != 'test':
            category_id = self.df["category_id"].iloc[idx]
        else:
            category_id = None  # For test set, no ground truth

        image = Image.open(file_path).convert('RGB')  # Ensure RGB format

        if self.transform:
            image = self.transform(image)

        return image, category_id, file_path, observation_id

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.df)

    def get_class_id(self, idx: int) -> int:
        """
        Returns the class ID of a specific sample.
        """
        if "category_id" in self.df.columns:
            return self.df["category_id"].iloc[idx]
        return None

    def show_sample(self, idx: int) -> None:
        """
        Displays a sample image along with its class name and index.
        """
        image, category_id, _, _, _ = self.__getitem__(idx)
        class_name = self.category_id2label[category_id] if category_id is not None else "Unknown"

        plt.imshow(image)
        plt.title(f"Class: {class_name}; id: {idx}")
        plt.axis('off')
        plt.show()

    def get_category_idxs(self, category_id: int) -> List[int]:
        """
        Retrieves all indexes for a given category ID.
        """
        if "category_id" in self.df.columns:
            return self.df[self.df.category_id == category_id].index.tolist()
        return []

class FungiEmbedder(nn.Module):
    """
    Wrapper for extracting image embeddings using a pre-trained visual model.
    Most layers are frozen except the final two transformer blocks.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        for name, param in self.model.named_parameters():
            if "visual.transformer.resblocks.11" not in name and "visual.transformer.resblocks.10" not in name:
                param.requires_grad = False

    def forward(self, pixel_values):
       return self.model.encode_image(pixel_values)

class PrototypicalLoss(nn.Module):
    """
    Prototypical loss that computes classification loss based on distances
    between input embeddings and class prototypes.
    """
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, targets, prototypes):
        dists = torch.cdist(embeddings, prototypes, p=2)
        logits = -dists
        loss = nn.functional.cross_entropy(logits, targets)
        return loss, logits

def compute_prototypes(model, dataloader, device, num_classes):
    """
    Compute class prototypes by averaging embeddings from support samples
    for each class in the dataset.
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
    """
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]  # will be all None
    file_paths = [item[2] for item in batch]
    observation_ids = [item[3] for item in batch]
    return images, labels, file_paths, observation_ids

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


def main():
    data_root = "data/fungi-clef-2025"

    # Load BioCLIP model and preprocessing function
    model, _, preprocess = open_clip.create_model_and_transforms(config.model_name)
    embedder = FungiEmbedder(model).to(config.device)
    
    # for name, param in embedder.named_parameters():
    #     if param.requires_grad:
    #         print(f"Training parameter: {name}")
    #     else:
    #         print(f"Freezing parameter: {name}")

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

    # Compute class prototypes from training data
    prototypes = compute_prototypes(embedder, train_loader, config.device, config.num_classes)

    # Training loop
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        train_loss, train_acc, train_f1 = train_epoch_protonet(embedder, train_loader, criterion, optimizer, config.device, prototypes)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=test_collate_fn
    )
    
    # Evaluate on test set
    evaluate_protonet_grouped(
        model=embedder,
        dataloader=test_loader,
        prototypes=prototypes,
        device=config.device,
        dataset=test_dataset,
        k=config.top_k,
        save_path="results/test_predictions_protoNet_grouped.csv"
    )

if __name__ == "__main__":
    main()
