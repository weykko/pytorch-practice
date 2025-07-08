import torch
from torchvision import transforms
from utils.datasets import load_dataset
from utils.utils import show_single_augmentation, show_multiple_augmentations


def create_augmentation_pipeline():
    """
    Создает и возвращает пайплайн аугментаций.
    """
    combined_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(200, padding=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.RandomGrayscale(p=0.3),
    ])

    augs = [
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
        ("RandomCrop", transforms.RandomCrop(200, padding=20)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
        ("RandomRotation", transforms.RandomRotation(degrees=30)),
        ("RandomGrayscale", transforms.RandomGrayscale(p=1.0)),
        ("CombinedAug", combined_aug)
    ]
    return augs


def process_and_visualize_augmentations(selected_images, selected_labels, augs, class_names):
    """
    Применяет аугментации к изображениям и визуализирует результаты.
    """
    to_tensor = transforms.ToTensor()

    for img_idx, (original_img, label) in enumerate(zip(selected_images, selected_labels)):
        original_tensor = to_tensor(original_img)
        all_aug_imgs = []

        for aug_name, aug in augs:
            aug_transform = transforms.Compose([
                aug,
                transforms.ToTensor()
            ])
            aug_img = aug_transform(original_img)
            all_aug_imgs.append(aug_img)
            show_single_augmentation(original_tensor, aug_img, f"plots/1_{class_names[label]}_{aug_name}.png", aug_name)

        titles = [a[0] for a in augs]
        show_multiple_augmentations(original_tensor, all_aug_imgs, titles, f"plots/1_{class_names[label]}_all.png")


def main():
    # Параметры
    root = 'data/train'
    target_size = (224, 224)

    # Загружаем датасет и выбираем изображения
    selected_images, selected_labels, class_names = load_dataset(root, target_size, 5)

    # Создаем пайплайн аугментаций
    augs = create_augmentation_pipeline()

    # Обрабатываем и визуализируем изображения с аугментациями
    process_and_visualize_augmentations(selected_images, selected_labels, augs, class_names)


if __name__ == "__main__":
    main()
