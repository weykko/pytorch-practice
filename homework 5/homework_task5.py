import os
import time
import psutil
from PIL import Image
import matplotlib.pyplot as plt
from utils.datasets import CustomImageDataset
from torchvision import transforms
from utils.extra_augs import AugmentationPipeline, CustomBrightnessContrast, CustomGaussianBlur


def load_dataset(root, num_images):
    """
    Загружает датасет и выбирает кол-во изображений для эксперимента.
    """
    dataset = CustomImageDataset(root, transform=None, target_size=None)
    selected_images = dataset.images[:num_images]
    return selected_images


def run_experiment(sizes, selected_images, pipeline):
    """
    Проводит эксперимент с разными размерами изображений.
    """
    times = []
    memories = []

    for size in sizes:
        print(f"\nЭксперимент для размера {size}")
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_start = process.memory_info().rss

        for img_path in selected_images:
            with Image.open(img_path) as img:
                img = img.resize(size, Image.Resampling.LANCZOS)
                aug_img = pipeline.apply(img)

        end_time = time.time()
        memory_end = process.memory_info().rss
        elapsed_time = end_time - start_time
        memory_used = memory_end - memory_start

        times.append(elapsed_time)
        memories.append(memory_used / (1024 * 1024))  # В мегабайтах
        print(f"Время: {elapsed_time:.2f} с")
        print(f"Память: {memory_used / (1024 * 1024):.2f} МБ")

    return times, memories


def plot_results(sizes, times, memories, path):
    """
    Строит графики зависимости времени и памяти от размера изображений.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot([size[0] for size in sizes], times, marker='o', color='orange')
    plt.title("Зависимость времени от размера изображений")
    plt.xlabel("Размер")
    plt.ylabel("Время (секунды)")

    plt.subplot(1, 2, 2)
    plt.plot([size[0] for size in sizes], memories, marker='o')
    plt.title("Зависимость памяти от размера изображений")
    plt.xlabel("Размер")
    plt.ylabel("Память (МБ)")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def main():
    root = './data/train'
    num_images = 100
    sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]

    # Загружаем датасет и выбираем изображения
    selected_images = load_dataset(root, num_images)

    # Настройка пайплайна аугментаций
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("HorizontalFlip", transforms.RandomHorizontalFlip(p=0.5), input_type='PIL')
    pipeline.add_augmentation("BrightnessContrast", CustomBrightnessContrast(p=0.7, brightness_factor=(0.8, 1.2), contrast_factor=(0.8, 1.2)), input_type='PIL')
    pipeline.add_augmentation("GaussianBlur", CustomGaussianBlur(p=0.5, max_radius=2), input_type='PIL')

    # Проводим эксперимент
    times, memories = run_experiment(sizes, selected_images, pipeline)

    # Строим графики
    plot_results(sizes, times, memories, "plots/5_plots.png")


if __name__ == "__main__":
    main()
