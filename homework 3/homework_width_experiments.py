import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.datasets import get_mnist_loaders
from utils.models import FullyConnectedModel
from utils.trainer import train_model
from utils.utils import count_parameters

# Устанавливаем устройство (GPU, если доступно, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем набор данных MNIST
train_loader, test_loader = get_mnist_loaders(batch_size=64)


def generate_layers(hidden_sizes):
    """
    Генерирует слои для модели на основе списка скрытых слоев.

    Аргументы:
        hidden_sizes (list): Список размеров скрытых слоев.

    Возвращает:
        list: Список слоев модели.
    """
    layers = []
    for size in hidden_sizes:
        layers.append({"type": "linear", "size": size})
        layers.append({"type": "relu"})
    return layers


def train_and_evaluate(layers, name):
    """
    Обучает модель с заданной конфигурацией слоев и оценивает её производительность.

    Аргументы:
        layers (list): Список слоев для модели.
        name (str): Имя модели.

    Возвращает:
        tuple: История обучения, количество параметров, время обучения, итоговая точность на тестовом наборе.
    """
    model = FullyConnectedModel(input_size=784, num_classes=10, layers=layers).to(device)
    param_count = count_parameters(model)  # Подсчет количества параметров
    start_time = time.time()
    history = train_model(model, train_loader, test_loader, epochs=10, device=device)
    end_time = time.time()
    training_time = end_time - start_time
    final_test_acc = history['test_accs'][-1]
    return history, param_count, training_time, final_test_acc


def experiment_width():
    """
    Эксперимент 2.1: Сравнение моделей с разной шириной слоев.

    Создает и обучает модели с разной шириной слоев, подсчитывает параметры и время обучения.

    Возвращает:
        dict: Истории обучения для каждой модели.
        dict: Количество параметров для каждой модели.
        dict: Время обучения для каждой модели.
        dict: Итоговая точность на тестовом наборе для каждой модели.
    """
    width_configs = {
        "narrow": [64, 32, 16],
        "medium": [256, 128, 64],
        "wide": [1024, 512, 256],
        "very_wide": [2048, 1024, 512]
    }

    histories = {}
    param_counts = {}
    training_times = {}
    final_test_accs = {}

    for name, hidden_sizes in width_configs.items():
        print(f"\nОбучение модели {name}")
        layers = generate_layers(hidden_sizes)
        history, param_count, training_time, final_test_acc = train_and_evaluate(layers, name)
        histories[name] = history
        param_counts[name] = param_count
        training_times[name] = training_time
        final_test_accs[name] = final_test_acc

    return histories, param_counts, training_times, final_test_accs


def plot_width_experiment_results(histories):
    """
    Построение heatmap для сравнения точности на тестовом наборе для разных моделей.

    Аргументы:
        final_test_accs (dict): Итоговая точность для каждой модели.
    """
    plt.figure(figsize=(10, 6))
    for name in histories:
        plt.plot(histories[name]['test_accs'], label=name)
    plt.xlabel("Эпоха")
    plt.ylabel("Точность на тестовом наборе")
    plt.legend()
    plt.title("Точность моделей с разной шириной")
    plt.tight_layout()
    plt.savefig("plots/width_experiment_results.png")
    plt.show()


def experiment_architecture_optimization():
    """
    Эксперимент 2.2: Оптимизация архитектуры с использованием сетчатого поиска.

    Исследует различные конфигурации архитектур и оценивает их точность на тестовом наборе.

    Возвращает:
        list: Результаты с точностями для каждой архитектуры.
    """
    arch_configs = [
        ([512, 256, 128], "contracting_512_256_128"),
        ([1024, 512, 256], "contracting_1024_512_256"),
        ([128, 256, 512], "expanding_128_256_512"),
        ([256, 512, 1024], "expanding_256_512_1024"),
        ([256, 256, 256], "constant_256"),
        ([512, 512, 512], "constant_512"),
        ([256, 128, 256], "bottleneck_256_128_256"),
        ([512, 256, 512], "bottleneck_512_256_512"),
    ]

    results = []
    for hidden_sizes, name in arch_configs:
        print(f"\nОбучение модели {name}...")
        layers = generate_layers(hidden_sizes)
        _, _, _, final_test_acc = train_and_evaluate(layers, name)
        results.append((name, final_test_acc))

    # Сортировка результатов по точности на тестовом наборе (по убыванию)
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def plot_architecture_comparison(results):
    """
    Построение heatmap для сравнения архитектур.

    Аргументы:
        results (list): Список результатов с точностями для каждой архитектуры.
    """
    architectures = [r[0] for r in results]
    accs = [r[1] for r in results]
    heatmap_data = np.array([accs]).reshape(1, -1)  # Сжимаем данные в одну строку
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, xticklabels=architectures, yticklabels=["Test Accuracy"], cmap="YlGnBu",
                cbar=True)
    plt.title('Сравнение точности различных архитектур')
    plt.xlabel('Архитектуры')
    plt.ylabel('Точность')
    plt.tight_layout()
    plt.savefig("plots/architecture_heatmap.png")
    plt.show()


def main():
    # Эксперимент 2.1: Сравнение моделей с разной шириной слоев
    # print("2.1: Сравнение моделей с разной шириной слоев")
    # histories, param_counts, training_times, final_test_accs = experiment_width()
    # plot_width_experiment_results(histories)
    # print("\nРезультаты экспериментов с шириной:")
    # for name in final_test_accs:
    #     print(
    #         f"{name}: Параметры: {param_counts[name]}, Время обучения: {training_times[name]:.2f} с, Итоговая точность test: {final_test_accs[name]:.4f}")

    # Эксперимент 2.2: Оптимизация архитектуры с использованием сетчатого поиска
    print("\n2.2: Оптимизация архитектуры с использованием сетчатого поиска")
    results = experiment_architecture_optimization()
    print("\nРезультаты:")
    for r in results:
        print(f"Архитектура: {r[0]}, Итоговая точность test: {r[1]}")
    plot_architecture_comparison(results)


if __name__ == '__main__':
    main()
