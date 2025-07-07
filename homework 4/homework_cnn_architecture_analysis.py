import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from utils.datasets import get_cifar_loaders
from utils.models import CNNKernelSize, CNNDepth
from utils.trainer import train_model
from utils.utils import plot_training_history, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_gradient_norms(model):
    """
    Функция для вычисления L2-нормы градиентов модели.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return np.sqrt(total_norm)


def compute_receptive_field(kernel_sizes, strides):
    """
    Функция для вычисления рецептивного поля.
    """
    rf = 1
    for k, s in zip(kernel_sizes, strides):
        rf += (k - 1) * s
    return rf


def visualize_activations(activations, title, path, num_filters=16):
    """
    Визуализирует активации (карты признаков).
    """
    activations = activations.detach().cpu().numpy()[0]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(activations[i], cmap='viridis')
            ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_comparison(histories, name1, name2, path,  metric='test_accs'):
    """
    Построение графиков для сравнения точности или потерь для двух моделей.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(histories[name1][metric], label=name1, marker='o')
    plt.plot(histories[name2][metric], label=name2, marker='s')
    plt.title(f'Сравнение {metric} на тестовой выборке')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(histories[name1]['test_losses'], label=name1, marker='o')
    plt.plot(histories[name2]['test_losses'], label=name2, marker='s')
    plt.title('Сравнение потерь на тестовой выборке')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_gradients_norms(gradient_norms, name, path):
    """
    Построение графиков норм градиентов во время обучения.
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


def train_and_evaluate(model, train_loader, test_loader, device):
    """
    Обучение модели и вычисление времени инференса.
    """
    start_time = time.time()
    history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=str(device))
    training_time = time.time() - start_time

    model.eval()
    inference_start = time.time()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            model(data)
    inference_time = (time.time() - inference_start) / len(test_loader.dataset)

    return history, training_time, inference_time


def analyze_kernel_sizes(train_loader, test_loader, device):
    """
    Анализ влияния размера ядра свертки на производительность.
    """
    kernel_configs = [
        ([3, 3, 3], "Ядра 3x3"),
        ([5, 5, 5], "Ядра 5x5"),
        ([7, 7, 7], "Ядра 7x7"),
        ([(1, 3), 3, 3], "Комбинация 1x1 + 3x3")
    ]

    histories_kernel = {}
    training_times_kernel = {}
    inference_times_kernel = {}
    receptive_fields = []

    for config, name in kernel_configs:
        print(f"\nОбучение CNN с {name}...")
        model = CNNKernelSize(config, input_channels=3, num_classes=10).to(device)

        # Обучаем модель
        history, training_time, inference_time = train_and_evaluate(model, train_loader, test_loader, device)

        histories_kernel[name] = history
        training_times_kernel[name] = training_time
        inference_times_kernel[name] = inference_time

        print(f"Параметры: {count_parameters(model)}")
        print(f"Время обучения: {training_time:.2f} с")
        print(f"Время инференса: {inference_time:.6f} с")

        kernel_sizes_for_rf = [k if isinstance(k, int) else 3 for k in config]
        rf = compute_receptive_field(kernel_sizes_for_rf, [1, 1, 1])
        receptive_fields.append((name, rf))
        print(f"Рецептивное поле: {rf}")

        # Визуализация активаций
        with torch.no_grad():
            for data, _ in test_loader:
                data = data[:1].to(device)
                activations = model.get_first_layer_activations(data)
                visualize_activations(activations, f"Активации первого слоя ({name})", f'plots/2_{name}_activations.png')
                break

        plot_training_history(history, f'plots/2_{name}_history.png')

    # Сравнение различных ядер
    plot_comparison(histories_kernel, 'Ядра 3x3', 'Ядра 5x5',f'plots/2_comp_3x3_5x5.png')
    plot_comparison(histories_kernel, 'Ядра 5x5', 'Ядра 7x7', f'plots/2_comp_5x5_7x7.png')
    plot_comparison(histories_kernel, 'Ядра 3x3', 'Комбинация 1x1 + 3x3', f'plots/2_comp_3x3_1x1+3x3.png')


def analyze_depth(train_loader, test_loader, device):
    """
    Анализ влияния глубины сети на производительность.
    """
    depth_configs = [
        (2, False, "Мелкая CNN (2 сверточных слоя)"),
        (4, False, "Средняя CNN (4 сверточных слоя)"),
        (6, False, "Глубокая CNN (6 сверточных слоев)"),
        (6, True, "Остаточная CNN (6 сверточных слоев)")
    ]

    histories_depth = {}
    training_times_depth = {}
    inference_times_depth = {}
    gradient_norms_depth = {}

    for num_layers, use_residual, name in depth_configs:
        print(f"\nОбучение {name}...")
        model = CNNDepth(num_layers, use_residual, input_channels=3, num_classes=10).to(device)

        gradient_norms = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()

        # Собираем нормы градиентов
        for epoch in range(5):
            epoch_grad_norms = []
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                epoch_grad_norms.append(compute_gradient_norms(model))
                optimizer.step()
            gradient_norms.append(np.mean(epoch_grad_norms))

        # Обучаем модель и получаем историю
        history, training_time, inference_time = train_and_evaluate(model, train_loader, test_loader, device)

        histories_depth[name] = history
        training_times_depth[name] = training_time
        inference_times_depth[name] = inference_time
        gradient_norms_depth[name] = gradient_norms

        print(f"Параметры: {count_parameters(model)}")
        print(f"Время обучения: {training_time:.2f} с")
        print(f"Время инференса: {inference_time:.6f} с")

        # Визуализация карт признаков
        with torch.no_grad():
            for data, _ in test_loader:
                data = data[:1].to(device)
                feature_maps = model.get_feature_maps(data, len(model.layers) - 4)
                visualize_activations(feature_maps, f"Карты признаков последнего слоя ({name})", f'plots/2_{name}_activations.png')
                break

        plot_training_history(history, f'plots/2_{name}_history.png')
        plot_gradients_norms(gradient_norms, name, f'plots/2_{name}_gradients.png')

    # Сравнение различных глубин
    plot_comparison(histories_depth, 'Мелкая CNN (2 сверточных слоя)', 'Средняя CNN (4 сверточных слоя)', f'plots/2_comp_2l_4l.png')
    plot_comparison(histories_depth, 'Средняя CNN (4 сверточных слоя)', 'Глубокая CNN (6 сверточных слоев)', f'plots/2_comp_4l_6l.png')
    plot_comparison(histories_depth, 'Глубокая CNN (6 сверточных слоев)', 'Остаточная CNN (6 сверточных слоев)', f'plots/2_comp_6l.png')


def main():
    train_loader, test_loader = get_cifar_loaders(batch_size=64)

    print("Влияние размера ядра свертки")
    analyze_kernel_sizes(train_loader, test_loader, device)

    print("\nВлияние глубины CNN")
    analyze_depth(train_loader, test_loader, device)


if __name__ == "__main__":
    main()
