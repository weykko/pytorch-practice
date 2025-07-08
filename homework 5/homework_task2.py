import torch
from torchvision import transforms
from utils.datasets import load_dataset
from utils.utils import show_single_augmentation, show_multiple_augmentations
from utils.extra_augs import AddGaussianNoise, RandomErasingCustom, AutoContrast
import random
from PIL import ImageEnhance, ImageFilter
import torchvision.transforms.functional as F


class CustomGaussianBlur:
    """Применяет случайное гауссово размытие к изображению."""

    def __init__(self, p=0.5, max_radius=3):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img):
        if random.random() > self.p:
            return img
        radius = random.uniform(0, self.max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class CustomPerspective:
    """Применяет случайное перспективное искажение."""

    def __init__(self, p=0.5, distortion_scale=0.3):
        self.p = p
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_tensor = F.to_tensor(img)
        img_pil = F.perspective(
            img,
            startpoints=[[0, 0], [img.width, 0], [img.width, img.height], [0, img.height]],
            endpoints=[
                [random.uniform(0, img.width * self.distortion_scale),
                 random.uniform(0, img.height * self.distortion_scale)],
                [img.width - random.uniform(0, img.width * self.distortion_scale),
                 random.uniform(0, img.height * self.distortion_scale)],
                [img.width - random.uniform(0, img.width * self.distortion_scale),
                 img.height - random.uniform(0, img.height * self.distortion_scale)],
                [random.uniform(0, img.width * self.distortion_scale),
                 img.height - random.uniform(0, img.height * self.distortion_scale)]
            ]
        )
        return img_pil


class CustomBrightnessContrast:
    """Случайно изменяет яркость и контрастность изображения."""

    def __init__(self, p=0.5, brightness_factor=(0.7, 1.3), contrast_factor=(0.7, 1.3)):
        self.p = p
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        if random.random() > self.p:
            return img
        brightness = random.uniform(*self.brightness_factor)
        contrast = random.uniform(*self.contrast_factor)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        return img


def create_augmentation_pipeline():
    """
    Создает и возвращает список кастомных аугментаций.
    """
    custom_augs = [
        ("CustomGaussianBlur", CustomGaussianBlur(p=1.0, max_radius=3)),
        ("CustomPerspective", CustomPerspective(p=1.0, distortion_scale=0.3)),
        ("CustomBrightnessContrast", CustomBrightnessContrast(p=1.0, brightness_factor=(0.7, 1.3), contrast_factor=(0.7, 1.3)))
    ]

    extra_augs = [
        ("AddGaussianNoise", AddGaussianNoise(mean=0., std=0.2)),
        ("RandomErasingCustom", RandomErasingCustom(p=1.0, scale=(0.02, 0.2))),
        ("AutoContrast", AutoContrast(p=1.0))
    ]

    return custom_augs, extra_augs


def process_and_visualize_augmentations(selected_images, selected_labels, custom_augs, extra_augs, class_names):
    """
    Применяет кастомные аугментации и сравнивает с готовыми аугментациями.
    """
    to_tensor = transforms.ToTensor()

    for img_idx, (original_img, label) in enumerate(zip(selected_images, selected_labels)):
        original_tensor = to_tensor(original_img)
        all_aug_imgs = []

        # Демонстрация кастомных аугментаций
        for aug_name, aug in custom_augs:
            aug_transform = transforms.Compose([aug, to_tensor])
            aug_img = aug_transform(original_img)
            all_aug_imgs.append(aug_img)
            show_single_augmentation(original_tensor, aug_img, f"plots/2_{class_names[label]}_{aug_name}.png", aug_name)

        # Демонстрация аугментаций из extra_augs
        for aug_name, aug in extra_augs:
            aug_transform = transforms.Compose([to_tensor, aug])
            aug_img = aug_transform(original_img)
            all_aug_imgs.append(aug_img)
            show_single_augmentation(original_tensor, aug_img, f"plots/2_{class_names[label]}_{aug_name}.png", aug_name)

        titles = [a[0] for a in custom_augs + extra_augs]
        show_multiple_augmentations(original_tensor, all_aug_imgs, titles, f"plots/2_{class_names[label]}_all.png")


def main():
    # Параметры
    root = 'data/train'
    target_size = (224, 224)

    # Загружаем датасет и выбираем изображения
    selected_images, selected_labels, class_names = load_dataset(root, target_size, 5)

    # Создаем список аугментаций
    custom_augs, extra_augs = create_augmentation_pipeline()

    # Обрабатываем и визуализируем изображения с аугментациями
    process_and_visualize_augmentations(selected_images, selected_labels, custom_augs, extra_augs, class_names)


if __name__ == "__main__":
    main()
