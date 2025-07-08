import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.datasets import CustomImageDataset


def load_dataset(root):
    """
    Загружает датасет без аугментаций, чтобы получить оригинальные размеры изображений.
    """
    dataset = CustomImageDataset(root, transform=None, target_size=None)
    class_names = dataset.get_class_names()
    return dataset, class_names


def count_images_per_class(root, class_names):
    """
    Подсчитывает количество изображений в каждом классе.
    """
    class_counts = {}
    for class_name in class_names:
        class_dir = os.path.join(root, class_name)
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        class_counts[class_name] = len(img_files)
    return class_counts


def analyze_image_sizes(dataset):
    """
    Анализирует размеры изображений в датасете.
    Возвращает минимальные, максимальные и средние размеры изображений.
    """
    widths = []
    heights = []
    for img_path in dataset.images:
        with Image.open(img_path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)

    min_width, max_width = min(widths), max(widths)
    min_height, max_height = min(heights), max(heights)
    mean_width, mean_height = np.mean(widths), np.mean(heights)

    return min_width, max_width, min_height, max_height, mean_width, mean_height, widths, heights


def visualize_size_distribution(widths, heights, path):
    """
    Визуализирует распределение размеров изображений на scatter plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5, c='blue', s=50)
    plt.title("Распределение размеров изображений")
    plt.xlabel("Ширина")
    plt.ylabel("Высота")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def visualize_class_histogram(class_counts, path):
    """
    Визуализирует гистограмму количества изображений по классам.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color='limegreen')
    plt.title("Гистограмма количества изображений по классам")
    plt.xlabel("Классы")
    plt.ylabel("Количество изображений")
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.25, int(yval), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def main():
    # Параметры
    root = 'data/train'

    # Загружаем датасет и получаем классы
    dataset, class_names = load_dataset(root)

    # Подсчитываем количество изображений в каждом классе
    class_counts = count_images_per_class(root, class_names)

    print("\nКоличество изображений в каждом классе")
    for class_name, count in class_counts.items():
        print(f"Класс {class_name}: {count} изображений")

    # Анализируем размеры изображений
    min_width, max_width, min_height, max_height, mean_width, mean_height, widths, heights = analyze_image_sizes(dataset)

    print("\nРазмеры изображений")
    print(f"Минимальный размер: {min_width}x{min_height}")
    print(f"Максимальный размер: {max_width}x{max_height}")
    print(f"Средний размер: {mean_width:.0f}x{mean_height:.0f}")

    # Визуализируем распределение размеров изображений
    visualize_size_distribution(widths, heights, "plots/3_img_distribution.png")

    # Визуализируем гистограмму по классам
    visualize_class_histogram(class_counts, "plots/3_count_img_by_class.png")


if __name__ == "__main__":
    main()
