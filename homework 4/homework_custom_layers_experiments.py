import time
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from utils.datasets import get_cifar_loaders
from utils.trainer import train_model
from utils.utils import plot_training_history, count_parameters
from utils.models import CIFARCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, noise_stddev=0.1):
        """
        Кастомный сверточный слой с добавлением гауссового шума.
        """
        super(CustomConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.noise_stddev = noise_stddev

    def forward(self, x):
        """
        Пропуск данных через свертку с добавлением шума.
        """
        x = self.conv(x)
        noise = torch.randn_like(x) * self.noise_stddev
        return x + noise


class ChannelAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Механизм внимания по каналам.
        """
        super(ChannelAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Применение внимания на основе средней и максимальной активации.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale_map = torch.cat([avg_out, max_out], dim=1)
        attn_mask = self.conv(scale_map)
        return x * self.sigmoid(attn_mask)


class CustomActivation(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Кастомная функция активации PELU.
        """
        super(CustomActivation, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        """
        Применение PELU.
        """
        positive_part = torch.where(x >= 0, x, torch.zeros_like(x))
        negative_part = torch.where(x < 0, self.alpha * (torch.exp(x / self.beta) - 1), torch.zeros_like(x))
        return positive_part + negative_part


class CustomPooling(nn.Module):
    def __init__(self):
        """
        Кастомный слой пулинга с усреднением и максимальным пулингом.
        """
        super(CustomPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(2, 2)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """
        Выполнение усреднённого и максимального пулинга.
        """
        avg_pooled = self.avgpool(x)
        max_pooled = self.maxpool(x)
        return 0.5 * (avg_pooled + max_pooled)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        """
        Основная модель с кастомными слоями.
        """
        super(CustomCNN, self).__init__()
        self.conv1 = CustomConvLayer(3, 32, kernel_size=3, padding=1)
        self.attn1 = ChannelAttention()
        self.act1 = CustomActivation()
        self.pool1 = CustomPooling()
        self.conv2 = CustomConvLayer(32, 64, kernel_size=3, padding=1)
        self.attn2 = ChannelAttention()
        self.act2 = CustomActivation()
        self.pool2 = CustomPooling()
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def compute_gradient_norms(model):
    """
    Функция для вычисления нормы градиентов.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2-норма градиента
            total_norm += param_norm.item() ** 2
    return np.sqrt(total_norm)


def run_experiment():
    """
    Основная функция для проведения экспериментов.
    """
    train_loader, test_loader = get_cifar_loaders(batch_size=64)

    # Проверка кастомных слоев
    custom_conv = CustomConvLayer(3, 64, kernel_size=3, padding=1)
    input_tensor = torch.randn(1, 3, 32, 32)
    print("Custom Conv:", custom_conv(input_tensor).shape)

    attention_layer = ChannelAttention()
    print("Attention:", attention_layer(input_tensor).shape)

    activation_layer = CustomActivation()
    print("Activation:", activation_layer(input_tensor).shape)

    pooling_layer = CustomPooling()
    print("Pooling:", pooling_layer(input_tensor).shape)

    # Инициализация и обучение моделей
    custom_cnn = CustomCNN()
    standard_cnn = CIFARCNN()

    models = {'CustomCNN': custom_cnn, 'StandardCNN': standard_cnn}
    histories = {}
    training_times = {}

    for name, model in models.items():
        print(f"\nОбучение {name}...")
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=str(device))
        training_time = time.time() - start_time

        histories[name] = history
        training_times[name] = training_time

        print(f"Параметры: {count_parameters(model)}")
        print(f"Время обучения: {training_time:.2f} секунд")
        plot_training_history(history, f"plots/3_{name}_history")  # Строим график истории обучения

        # Сравнение пользовательской и стандартной CNN
        compare_models(histories)


def compare_models(histories):
    """
    Функция для сравнения двух моделей.
    """
    plt.figure(figsize=(12, 4))

    # Сравнение точности
    plt.subplot(1, 2, 1)
    plt.plot(histories['CustomCNN']['test_accs'], label='Custom CNN', marker='o')
    plt.plot(histories['StandardCNN']['test_accs'], label='Standard CNN', marker='s')
    plt.title('Сравнение точности на тестовой выборке')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/3_comp_custom_standard_loss.png")
    plt.grid(True)

    # Сравнение потерь
    plt.subplot(1, 2, 2)
    plt.plot(histories['CustomCNN']['test_losses'], label='Custom CNN', marker='o')
    plt.plot(histories['StandardCNN']['test_losses'], label='Standard CNN', marker='s')
    plt.title('Сравнение потерь на тестовой выборке')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/3_comp_custom_standard_acc.png")
    plt.show()


if __name__ == "__main__":
    run_experiment()
