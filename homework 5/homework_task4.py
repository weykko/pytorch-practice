import torch
from torchvision import transforms
from utils.extra_augs import CustomPerspective, CustomBrightnessContrast, CustomGaussianBlur
from PIL import Image
from utils.datasets import CustomImageDataset, load_dataset


class AugmentationPipeline:
    """
    Класс для управления пайплайном аугментаций изображений.
    """

    def __init__(self):
        self.augmentations = {}  # Словарь: {name: (aug, input_type)}
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def add_augmentation(self, name, aug, input_type='PIL'):
        """
        Добавляет аугментацию в пайплайн.
        """
        if not isinstance(name, str):
            raise ValueError("Имя аугментации должно быть строкой")
        if input_type not in ['PIL', 'tensor']:
            raise ValueError("input_type должен быть 'PIL' или 'tensor'")
        self.augmentations[name] = (aug, input_type)

    def remove_augmentation(self, name):
        """
        Удаляет аугментацию из пайплайна по имени.
        """
        if name in self.augmentations:
            del self.augmentations[name]
        else:
            raise KeyError(f"Аугментация с именем '{name}' не найдена")

    def apply(self, image):
        """
        Применяет все аугментации к изображению.
        """
        result = image
        for name, (aug, input_type) in self.augmentations.items():
            if input_type == 'tensor' and isinstance(result, Image.Image):
                result = self.to_tensor(result)
            elif input_type == 'PIL' and isinstance(result, torch.Tensor):
                result = self.to_pil(result)
            result = aug(result)
        return result

    def get_augmentations(self):
        """
        Возвращает список всех аугментаций в пайплайне.
        """
        return list(self.augmentations.keys())


def create_augmentation_pipelines():
    """
    Создает различные конфигурации пайплайнов аугментаций.
    """
    # Light конфигурация
    light_pipeline = AugmentationPipeline()
    light_pipeline.add_augmentation("HorizontalFlip", transforms.RandomHorizontalFlip(p=0.5), input_type='PIL')
    light_pipeline.add_augmentation("Crop", transforms.RandomCrop(200, padding=20, padding_mode='reflect'))
    light_pipeline.add_augmentation("BrightnessContrast", CustomBrightnessContrast(p=0.5, brightness_factor=(0.9, 1.1), contrast_factor=(0.9, 1.1)), input_type='PIL')

    # Medium конфигурация
    medium_pipeline = AugmentationPipeline()
    medium_pipeline.add_augmentation("HorizontalFlip", transforms.RandomHorizontalFlip(p=0.5), input_type='PIL')
    medium_pipeline.add_augmentation("Crop", transforms.RandomCrop(200, padding=20, padding_mode='reflect'))
    medium_pipeline.add_augmentation("ColorJitter",transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    medium_pipeline.add_augmentation("BrightnessContrast", CustomBrightnessContrast(p=0.7, brightness_factor=(0.8, 1.2), contrast_factor=(0.8, 1.2)), input_type='PIL')
    medium_pipeline.add_augmentation("GaussianBlur", CustomGaussianBlur(p=0.9, max_radius=10), input_type='PIL')

    # Heavy конфигурация
    heavy_pipeline = AugmentationPipeline()
    heavy_pipeline.add_augmentation("HorizontalFlip", transforms.RandomHorizontalFlip(p=0.8), input_type='PIL')
    heavy_pipeline.add_augmentation("Crop", transforms.RandomCrop(200, padding=20, padding_mode='reflect'))
    heavy_pipeline.add_augmentation("ColorJitter",transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    heavy_pipeline.add_augmentation("BrightnessContrast", CustomBrightnessContrast(p=0.8, brightness_factor=(0.7, 1.3), contrast_factor=(0.7, 1.3)), input_type='PIL')
    heavy_pipeline.add_augmentation("GaussianBlur", CustomGaussianBlur(p=0.8, max_radius=3), input_type='PIL')
    heavy_pipeline.add_augmentation("Perspective", CustomPerspective(p=0.8, distortion_scale=0.4), input_type='PIL')
    heavy_pipeline.add_augmentation("Rotation", transforms.RandomRotation(degrees=45))

    return light_pipeline, medium_pipeline, heavy_pipeline


def save_augmented_images(selected_images, selected_labels, pipelines, class_names):
    """
    Применяет аугментации и сохраняет результат.
    """
    for pipeline_name, pipeline in pipelines:
        for img_idx, (img, label) in enumerate(zip(selected_images, selected_labels)):
            aug_img = pipeline.apply(img)
            aug_img.save(f"plots/4_{pipeline_name}_{class_names[label]}.jpg")


def main():
    root = 'data/train'
    target_size = (224, 224)
    # Загружаем датасет
    selected_images, selected_labels, class_names = load_dataset(root, target_size, 3)

    # Создаем конфигурации пайплайнов
    light_pipeline, medium_pipeline, heavy_pipeline = create_augmentation_pipelines()

    # Список конфигураций
    pipelines = [
        ("light", light_pipeline),
        ("medium", medium_pipeline),
        ("heavy", heavy_pipeline)
    ]

    # Применение и сохранение аугментаций
    save_augmented_images(selected_images, selected_labels, pipelines, class_names)


if __name__ == "__main__":
    main()
