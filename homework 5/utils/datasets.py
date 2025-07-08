import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CustomImageDataset(Dataset):
    """Кастомный датасет для работы с папками классов"""

    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir (str): Путь к папке с классами
            transform: Аугментации для изображений
            target_size (tuple): Размер для ресайза изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        # Получаем список классов (папок)
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Собираем все пути к изображениям
        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')

        # Ресайзим изображение
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Применяем аугментации
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes


def load_dataset(root, target_size, num_images):
    """
    Загружает датасет и выбирает по одному изображению из разных классов.
    """
    dataset = CustomImageDataset(root, transform=None, target_size=target_size)
    class_names = dataset.get_class_names()

    selected_images = []
    selected_labels = []
    class_indices = {}

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label not in class_indices and len(class_indices) < num_images:
            class_indices[label] = idx
            selected_images.append(img)
            selected_labels.append(label)
        if len(class_indices) == num_images:
            break

    return selected_images, selected_labels, class_names