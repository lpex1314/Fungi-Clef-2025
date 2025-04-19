import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt
from typing import List

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
        for name, param in model.named_parameters():
            if "visual.transformer.resblocks." in name and any(f"resblocks.{i}" in name for i in [9, 10, 11]):
                param.requires_grad = True
            elif "ln_post" in name or "positional_embedding" or "visual.proj" in name:
                param.requires_grad = True
            else:
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
        dists = torch.cdist(embeddings, prototypes, p=2) # (B, C)
        logits = -dists # (B, C)
        loss = nn.functional.cross_entropy(logits, targets)
        return loss, logits
