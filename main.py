import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from typing import List
import matplotlib.pyplot as plt
import kagglehub
import open_clip  # Import open_clip instead of bioclip

# 设置随机种子以保证可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# 配置参数
class Config:
    batch_size = 32
    learning_rate = 2e-5
    epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "hf-hub:imageomics/bioclip"  # 从Hugging Face Hub加载BioCLIP模型
    num_classes = 0  # 将在加载数据集后更新
    image_size = 224  # BioCLIP预期的图像大小
    top_k = 10  # 评估前k个预测类别

config = Config()

# 使用提供的FungiTastic类
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

# 针对大量类别优化的分类器
class FungiClassifier(nn.Module):
    def __init__(self, model, preprocess, num_classes):
        super(FungiClassifier, self).__init__()
        # 保存预训练模型
        self.model = model
        self.preprocess = preprocess
        
        # 冻结模型的大部分参数
        for name, param in self.model.named_parameters():
            if "visual.transformer.resblocks.11" not in name and "visual.transformer.resblocks.10" not in name:
                param.requires_grad = False
        
        # 获取BioCLIP的嵌入维度
        if hasattr(self.model, 'visual.proj'):
            embedding_dim = self.model.visual.proj.shape[0]  # CLIP-style models
        else:
            # Typically 512 for BioCLIP
            embedding_dim = 512
        
        # 针对大量类别优化的分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )
        
        # 初始化分类器权重
        self._init_weights(self.classifier)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, pixel_values):
        # 获取特征
        with torch.no_grad():
            image_features = self.model.encode_image(pixel_values)
            
        # 通过分类器
        logits = self.classifier(image_features)
        return logits

# 标签平滑交叉熵损失
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        images, labels, _, _ = batch  # 忽略其他信息
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累积损失
        epoch_loss += loss.item()
        
        # 收集预测结果
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')  # 对于多类别，使用macro平均
    
    return epoch_loss / len(dataloader), accuracy, f1

# 评估函数 - 包含Top-K预测
def evaluate(model, dataloader, criterion, device, dataset, k=10):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 存储所有预测的概率
    all_observation_ids = []  # 存储所有观察ID
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, labels, _, observation_ids = batch
            images = images.to(device)
            
            if labels is not None:
                labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 使用softmax获取类别概率
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            if labels is not None:
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                
                # 收集预测结果
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            all_probs.extend(probs.cpu().numpy())
            all_observation_ids.extend(observation_ids)
    
    # 如果有标签，计算标准指标
    if all_labels:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        top_k_acc = top_k_accuracy_score(all_labels, all_probs, k=k, labels=range(dataset.n_classes))
    else:
        accuracy = f1 = top_k_acc = 0.0
    
    # 创建预测结果
    results = []
    for i, (obs_id, probs) in enumerate(zip(all_observation_ids, all_probs)):
        # 获取前k个最高概率的索引
        top_k_indices = np.argsort(probs)[-k:][::-1]
        
        # 创建以空格分隔的前k个预测的类别ID字符串
        predictions = ' '.join([str(idx) for idx in top_k_indices])
        
        result = {
            "ObservationId": obs_id,
            "predictions": predictions
        }
        
        results.append(result)
    
    # 保存为CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("test_predictions.csv", index=False)
    
    if all_labels:
        return epoch_loss / len(dataloader), accuracy, f1, top_k_acc
    else:
        return 0, 0, 0, 0

# 主函数
def main():
    # 路径设置
    data_root = kagglehub.competition_download(handle='fungi-clef-2025')  # 修改为实际数据集路径
    
    # 加载 BioCLIP 模型和预处理
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    # tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    
    # 创建数据集
    train_fungi_dataset = FungiTastic(
        root=data_root,
        split='train',
        transform=preprocess
    )
    
    val_fungi_dataset = FungiTastic(
        root=data_root,
        split='val',
        transform=preprocess
    )
    
    test_fungi_dataset = FungiTastic(
        root=data_root,
        split='test',
        transform=preprocess
    )
    
    # 更新配置中的类别数量
    config.num_classes = train_fungi_dataset.n_classes
    print(f"数据集包含 {config.num_classes} 个真菌类别")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_fungi_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_fungi_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_fungi_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = FungiClassifier(model, preprocess, config.num_classes)
    model = model.to(config.device)
    
    # 使用标签平滑的损失函数
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # 优化器 - 针对不同参数组使用不同学习率
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': config.learning_rate / 10},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': config.learning_rate}
    ])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=1,
        eta_min=1e-6
    )
    
    # 训练循环
    best_top_k = 0.0
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        # 评估
        val_loss, val_acc, val_f1, val_top_k = evaluate(
            model, val_loader, criterion, config.device, val_fungi_dataset, k=config.top_k
        )
        
        # 学习率调整
        scheduler.step()
        
        # 打印指标
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Top-{config.top_k} Acc: {val_top_k:.4f}")
        
        # 保存最佳模型 - 根据Top-K准确率
        if val_top_k > best_top_k:
            best_top_k = val_top_k
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_top_k': val_top_k,
                'category_id2label': val_fungi_dataset.category_id2label
            }, os.path.join(data_root, "best_fungi_classifier.pth"))
            print(f"保存最佳模型!Top-{config.top_k} Acc: {val_top_k:.4f}")
    
    print(f"训练完成！最佳 Top-{config.top_k} 准确率: {best_top_k:.4f}")
    
    # 加载最佳模型并进行最终评估
    checkpoint = torch.load(os.path.join(data_root, "best_fungi_classifier.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("使用最佳模型进行最终测试集评估...")
    evaluate(
        model, test_loader, criterion, config.device, test_fungi_dataset, k=config.top_k
    )
    
    print("测试集预测完成，结果已保存到 'test_predictions.csv'")

if __name__ == "__main__":
    main()