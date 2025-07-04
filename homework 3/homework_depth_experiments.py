import torch
import time
from utils.datasets import get_mnist_loaders
from utils.models import FullyConnectedModel
from utils.trainer import train_model
from utils.utils import plot_training_history

# Определяем устройство для обучения (GPU, если доступно, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем набор данных MNIST
train_loader, test_loader = get_mnist_loaders(batch_size=64)


def train_and_evaluate(models_dict, train_loader, test_loader, epochs=10):
    """
    Функция для обучения и оценки моделей на тренировочных и тестовых данных.

    Аргументы:
        models_dict (dict): Словарь с моделями для обучения.
        train_loader (DataLoader): Загрузчик для обучающих данных.
        test_loader (DataLoader): Загрузчик для тестовых данных.
        epochs (int): Количество эпох для обучения модели.

    Возвращает:
        dict: Истории обучения для каждой модели.
        dict: Время обучения для каждой модели.
    """
    training_histories = {}
    training_times = {}

    for model_name, model in models_dict.items():
        print(f"\nОбучение модели {model_name}")
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
        end_time = time.time()

        elapsed_time = end_time - start_time
        training_histories[model_name] = history
        training_times[model_name] = elapsed_time

        print(f"{model_name} - Время обучения: {elapsed_time:.2f} секунд")
        print(f"{model_name} - Итоговая точность на тестовом наборе: {history['test_accs'][-1]:.4f}")

        plot_training_history(history, f"plots/{model_name}.png")

    return training_histories, training_times


def experiment_with_model_depth():
    """
    Функция для создания и обучения моделей с разной глубиной.

    Создает модели с 1, 2, 3, 5 и 7 слоями и обучает их.

    Возвращает:
        dict: Истории обучения для каждой модели.
        dict: Время обучения для каждой модели.
    """
    # Определяем модели с разной глубиной
    models_dict = {
        "1_layer": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[]
        ).to(device),
        "2_layers": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[{"type": "linear", "size": 256}, {"type": "relu"}]
        ).to(device),
        "3_layers": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[{"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"}]
        ).to(device),
        "5_layers": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[{"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"}]
        ).to(device),
        "7_layers": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[{"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"},
                    {"type": "linear", "size": 256}, {"type": "relu"}]
        ).to(device)
    }

    model_histories, model_times = train_and_evaluate(models_dict, train_loader, test_loader)
    return model_histories, model_times


def experiment_with_regularization():
    """
    Функция для создания и обучения моделей с регуляризацией (Dropout и BatchNorm).

    Создает модели с 2, 3, 5 и 7 слоями и применяет регуляризацию, такую как BatchNorm и Dropout.

    Возвращает:
        dict: Истории обучения для каждой модели.
        dict: Время обучения для каждой модели.
    """
    # Модели с регуляризацией
    models_with_reg = {
        "2_layers_reg": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[{"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2}]
        ).to(device),
        "3_layers_reg": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[{"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2}]
        ).to(device),
        "5_layers_reg": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[{"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2}]
        ).to(device),
        "7_layers_reg": FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[{"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2},
                    {"type": "linear", "size": 256}, {"type": "batch_norm"},
                    {"type": "relu"}, {"type": "dropout", "rate": 0.2}]
        ).to(device)
    }

    print("\n1.2: Анализ переобучения с регуляризацией")
    reg_histories, reg_times = train_and_evaluate(models_with_reg, train_loader, test_loader)
    return reg_histories, reg_times


def analyze_optimal_depth(base_histories, reg_histories):
    """
    Функция для анализа и вывода оптимальной глубины модели, а также сравнительного анализа точности.

    Аргументы:
        base_histories (dict): Истории обучения для базовых моделей.
        reg_histories (dict): Истории обучения для моделей с регуляризацией.
    """
    print("\nСводный анализ")

    print("\nБазовые модели:")
    for model_name, history in base_histories.items():
        print(f"{model_name}:")
        print(f"Итоговая точность train: {history['train_accs'][-1]:.4f}")
        print(f"Итоговая точность test: {history['test_accs'][-1]:.4f}")

    print("\nМодели с регуляризацией:")
    for model_name, history in reg_histories.items():
        print(f"{model_name}:")
        print(f"Итоговая точность train: {history['train_accs'][-1]:.4f}")
        print(f"Итоговая точность test: {history['test_accs'][-1]:.4f}")

    # Анализ оптимальной глубины
    best_test_acc = 0
    best_model = ""
    for model_name, history in base_histories.items():
        test_acc = history['test_accs'][-1]
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = model_name
    print(f"\nЛучшая базовая модель: {best_model} с точностью на тестовом наборе: {best_test_acc:.4f}")

    best_test_acc_reg = 0
    best_model_reg = ""
    for model_name, history in reg_histories.items():
        test_acc = history['test_accs'][-1]
        if test_acc > best_test_acc_reg:
            best_test_acc_reg = test_acc
            best_model_reg = model_name
    print(f"Лучшая модель с регуляризацией: {best_model_reg} с точностью на тестовом наборе: {best_test_acc_reg:.4f}")


def main():
    print("1.1: Сравнение моделей разной глубины")
    base_histories, base_times = experiment_with_model_depth()

    print("\n1.2: Анализ переобучения с регуляризацией")
    reg_histories, reg_times = experiment_with_regularization()

    analyze_optimal_depth(base_histories, reg_histories)


if __name__ == '__main__':
    main()
