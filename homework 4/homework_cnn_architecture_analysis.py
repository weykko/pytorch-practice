import torch
import torch.nn as nn
from utils.datasets import get_mnist_loaders, get_cifar_loaders
from utils.models import SimpleCNN, CNNWithResidual, FullyConnectedNet, RegularizedCNNWithResidual
from utils.trainer import train_model
from utils.utils import plot_training_history, count_parameters, compare_models
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_confusion_matrix(model, data_loader, device):
    """
    Функция для вычисления и возвращения матрицы ошибок.
    """
    model.eval()  # Переводим модель в режим оценки
    all_preds = []
    all_targets = []
    with torch.no_grad():  # Отключаем вычисление градиентов
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)  # Получаем предсказанные классы
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return confusion_matrix(all_targets, all_preds)  # Возвращаем матрицу ошибок


def compute_gradient_norms(model):
    """
    Функция для вычисления L2-нормы градиентов модели.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2-норма градиента параметра
            total_norm += param_norm.item() ** 2
    return np.sqrt(total_norm)  # Возвращаем общую норму градиентов


def plot_confusion_matrix(cm, title, classes, path):
    """
    Визуализирует матрицу ошибок в виде тепловой карты.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_gradient_norms(gradient_norms, name, path):
    """
    Визуализирует график нормы градиентов во время обучения.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(gradient_norms, label=f'Норма градиентов {name}')
    plt.title(f'Нормы градиентов во время обучения ({name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Норма градиентов')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def run_experiment(model_class, dataset, epochs=5, lr=0.001, batch_size=64):
    """
    Обучение модели и вывод информации о производительности.
    """
    train_loader, test_loader = dataset(batch_size)
    model = model_class.to(device)

    # Обучение модели
    start_time = time.time()
    history = train_model(model, train_loader, test_loader, epochs, lr, device)
    training_time = time.time() - start_time

    # Измерение времени инференса
    model.eval()
    inference_start = time.time()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            model(data)
    inference_time = (time.time() - inference_start) / len(test_loader.dataset)

    # Вывод параметров и времени обучения
    print(f"Параметры: {count_parameters(model)}")
    print(f"Время обучения: {training_time:.2f} с")
    print(f"Время инференса: {inference_time:.6f} с")

    return model, history, training_time, inference_time


def experiment_mnist():
    """
    Эксперимент с использованием MNIST для сравнения различных моделей.
    """
    train_loader_mnist, test_loader_mnist = get_mnist_loaders(batch_size=64)

    models_mnist = {
        'FullyConnected': FullyConnectedNet(input_size=28 * 28, num_classes=10),
        'SimpleCNN': SimpleCNN(input_channels=1, num_classes=10),
        'ResidualCNN': CNNWithResidual(input_channels=1, num_classes=10)
    }

    histories_mnist = {}
    training_times_mnist = {}
    inference_times_mnist = {}

    for name, model in models_mnist.items():
        print(f"\nОбучение {name}")
        model, history, training_time, inference_time = run_experiment(
            model_class=models_mnist[name],
            dataset=get_mnist_loaders,
        )
        histories_mnist[name] = history
        training_times_mnist[name] = training_time
        inference_times_mnist[name] = inference_time

        plot_training_history(history, f'plots/{name}_history_mnist.png')  # Построение графика истории обучения

        cm = get_confusion_matrix(model, test_loader_mnist, device)
        plot_confusion_matrix(cm, f"Матрица ошибок {name} (MNIST)", range(10), f'plots/{name}_matrix_mnist.png')

    # Сравнение моделей
    compare_models(histories_mnist['FullyConnected'], histories_mnist['SimpleCNN'], f'plots/compare_fcn_cnn_mnist.png')
    compare_models(histories_mnist['SimpleCNN'], histories_mnist['ResidualCNN'], f'plots/compare_cnn_residual_cnn_mnist.png')


def experiment_cifar10():
    """
    Эксперимент с использованием CIFAR-10 для сравнения различных моделей.
    """
    train_loader_cifar, test_loader_cifar = get_cifar_loaders(batch_size=64)

    models_cifar = {
        'FullyConnected': FullyConnectedNet(input_size=32 * 32 * 3, num_classes=10),
        'ResidualCNN': CNNWithResidual(input_channels=3, num_classes=10),
        'RegularizedResidualCNN': RegularizedCNNWithResidual(input_channels=3, num_classes=10)
    }

    histories_cifar = {}
    training_times_cifar = {}
    inference_times_cifar = {}
    gradient_norms_cifar = {}

    for name, model in models_cifar.items():
        print(f"\nОбучение {name}...")
        model, history, training_time, inference_time = run_experiment(
            model_class=models_cifar[name],
            dataset=get_cifar_loaders,
        )
        histories_cifar[name] = history
        training_times_cifar[name] = training_time
        inference_times_cifar[name] = inference_time

        # Собираем нормы градиентов
        gradient_norms_cifar[name] = []
        for epoch in range(5):
            epoch_grad_norms = []
            for data, target in train_loader_cifar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                epoch_grad_norms.append(compute_gradient_norms(model))
                clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_norms_cifar[name].append(np.mean(epoch_grad_norms))

        plot_gradient_norms(gradient_norms_cifar[name], name, f'plots/{name}_gradient_cifar.png')
        plot_training_history(history, f'plots/{name}_history_cifar.png')

        cm = get_confusion_matrix(model, test_loader_cifar, device)
        cifar_classes = ['самолет', 'автомобиль', 'птица', 'кошка', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль',
                         'грузовик']
        plot_confusion_matrix(cm, f"Матрица ошибок {name} (CIFAR-10)", cifar_classes, f'plots/{name}_matrix_cifar.png')

    # Сравнение моделей
    compare_models(histories_cifar['FullyConnected'], histories_cifar['ResidualCNN'], f'plots/compare_fcn_rcnn_cifar.png')
    compare_models(histories_cifar['ResidualCNN'], histories_cifar['RegularizedResidualCNN'], f'plots/compare_rcnn_regularcnn_cifar.png')


def main():
    print("Сравнение на MNIST")
    experiment_mnist()
    print("\nСравнение на CIFAR-10")
    experiment_cifar10()


if __name__ == "__main__":
    main()
