import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from utils.datasets import get_cifar_loaders
from utils.trainer import train_model
from utils.utils import plot_training_history, count_parameters
from utils.models import CIFARCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, noise_stddev=0.1):
        super(CustomConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.noise_stddev = noise_stddev

    def forward(self, x):
        # Сворачиваем обычное изображение
        x = self.conv(x)
        # Добавляем гауссов шум
        noise = torch.randn_like(x) * self.noise_stddev
        noisy_output = x + noise
        return noisy_output

class ChannelAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ChannelAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Получаем среднюю и максимальную активации по каналам
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale_map = torch.cat([avg_out, max_out], dim=1)
        attn_mask = self.conv(scale_map)
        attn_mask = self.sigmoid(attn_mask)
        return x * attn_mask

# PELU
class CustomActivation(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CustomActivation, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        positive_part = torch.where(x >= 0, x, torch.zeros_like(x))
        negative_part = torch.where(x < 0, self.alpha * (torch.exp(x / self.beta) - 1), torch.zeros_like(x))
        return positive_part + negative_part

class CustomPooling(nn.Module):
    def __init__(self):
        super(CustomPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(2, 2)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        avg_pooled = self.avgpool(x)
        max_pooled = self.maxpool(x)
        hybrid_pooled = 0.5 * (avg_pooled + max_pooled)
        return hybrid_pooled

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
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
        x = self.fc(x)
        return x


# Базовый остаточный блок (Basic Residual Block)
class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)  # Первая свертка
        self.bn1 = nn.BatchNorm2d(out_channels)  # Пакетная нормализация
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)  # Вторая свертка
        self.bn2 = nn.BatchNorm2d(out_channels)  # Пакетная нормализация
        self.shortcut = nn.Sequential()  # Ярлык (ветка пропуска)
        # Если размеры не совпадают, используем свертку 1x1 для ярлыка
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # Свертка -> BatchNorm -> ReLU
        out = self.bn2(self.conv2(out))  # Свертка -> BatchNorm
        out += self.shortcut(x)  # Добавляем ярлык
        out = F.relu(out)  # ReLU
        return out


# Остаточный блок "бутылочное горлышко" (Bottleneck Residual Block)
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        bottleneck_channels = out_channels // 4  # Уменьшение количества каналов
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)  # Свертка 1x1
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, stride, 1, bias=False)  # Свертка 3x3
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1,
                               bias=False)  # Свертка 1x1 для восстановления каналов
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Широкий остаточный блок (Wide Residual Block)
class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, width_factor=2):
        super().__init__()
        wide_channels = out_channels * width_factor  # Увеличение количества каналов
        self.conv1 = nn.Conv2d(in_channels, wide_channels, 3, stride, 1, bias=False)  # Первая свертка
        self.bn1 = nn.BatchNorm2d(wide_channels)
        self.conv2 = nn.Conv2d(wide_channels, out_channels, 3, 1, 1, bias=False)  # Вторая свертка
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# CNN с остаточными блоками
class ResidualCNN(nn.Module):
    def __init__(self, block_type, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)  # Первый сверточный слой
        self.bn1 = nn.BatchNorm2d(32)  # Пакетная нормализация
        self.block1 = block_type(32, 32)  # Первый остаточный блок
        self.block2 = block_type(32, 64, stride=2)  # Второй остаточный блок с уменьшением размера
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Адаптивный усредняющий пулинг
        self.fc = nn.Linear(64 * 4 * 4, num_classes)  # Полносвязный слой

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Выпрямляем тензор
        x = self.fc(x)  # Полносвязный слой
        return x


# Вычисление нормы градиентов
def compute_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2-норма градиента
            total_norm += param_norm.item() ** 2
    return np.sqrt(total_norm)  # Возвращаем общую норму

def main():
    # Проверка кастомного свёрточного слоя
    custom_conv = CustomConvLayer(3, 64, kernel_size=3, padding=1)
    input_tensor = torch.randn(1, 3, 32, 32)
    output = custom_conv(input_tensor)
    print("Custom Conv:", output.shape)

    # Проверка механизма Attention
    spatial_attn = ChannelAttention()
    output_attention = spatial_attn(input_tensor)
    print("Spatial Attention:", output_attention.shape)

    # Проверка кастомной функции активации
    pelu_activation = CustomActivation()
    activated_tensor = pelu_activation(input_tensor)
    print("Custom Activated Tensor:", activated_tensor.shape)

    # Проверка кастомного пулинга
    hybrid_pool = CustomPooling()
    pooled_tensor = hybrid_pool(input_tensor)
    print("Custom Pooling:", pooled_tensor.shape)

    # Получаем загрузчики данных
    train_loader, test_loader = get_cifar_loaders(batch_size=64)

    # Тестируем обе модели
    custom_cnn = CustomCNN()
    standard_cnn = CIFARCNN()

    models = {'CustomCNN': custom_cnn, 'StandardCNN': standard_cnn}
    histories = {}
    training_times = {}

    for name, model in models.items():
        print(f"\nОбучение {name}...")
        start_time = time.time()
        # Обучаем модель и получаем историю
        history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=str(device))
        training_time = time.time() - start_time

        histories[name] = history
        training_times[name] = training_time

        print(f"Параметры: {count_parameters(model)}")
        print(f"Время обучения: {training_time:.2f} секунд")
        plot_training_history(history, f"plots/3_{name}_history")  # Строим график истории обучения

        # Сравнение пользовательской и стандартной CNN
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(histories['CustomCNN']['test_accs'], label='Custom CNN', marker='o')
        plt.plot(histories['StandardCNN']['test_accs'], label='Standard CNN', marker='s')
        plt.title('Сравнение точности на тестовой выборке')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/3_comp_custom_standard_loss.png")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(histories['CustomCNN']['test_losses'], label='Custom CNN', marker='o')
        plt.plot(histories['StandardCNN']['test_losses'], label='Standard CNN', marker='s')
        plt.title('Сравнение потерь на тестовой выборке')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/3_comp_custom_standard_acc.png")
        plt.show()

    # 3.2 Эксперимент с остаточными блоками
    print("\n=== Эксперимент с остаточными блоками ===")
    block_types = [
        (BasicResidualBlock, "Базовый остаточный"),
        (BottleneckResidualBlock, "Остаточный (бутылочное горлышко)"),
        (WideResidualBlock, "Широкий остаточный")
    ]

    histories_res = {}
    training_times_res = {}
    gradient_norms_res = {}

    for block_type, name in block_types:
        print(f"\nОбучение {name}...")
        model = ResidualCNN(block_type, input_channels=3, num_classes=10).to(device)

        # Отслеживание градиентов
        gradient_norms = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()

        for epoch in range(5):  # Выполняем 5 эпох для сбора норм градиентов
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

        start_time = time.time()
        # Обучаем модель и получаем историю
        history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=str(device))
        training_time = time.time() - start_time

        histories_res[name] = history
        training_times_res[name] = training_time
        gradient_norms_res[name] = gradient_norms  # Сохраняем нормы градиентов

        print(f"Параметры: {count_parameters(model)}")
        print(f"Время обучения: {training_time:.2f} секунд")
        plot_training_history(history, f"plots/3_{name}_history.png")

        # Построение графика норм градиентов
        plt.figure(figsize=(8, 4))
        plt.plot(gradient_norms, label=f'Норма градиентов {name}')
        plt.title(f'Нормы градиентов во время обучения ({name})')
        plt.xlabel('Эпоха')
        plt.ylabel('Норма градиентов')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/3_gradients_norms.png")
        plt.show()

    # Сравнение остаточных блоков
    for name1, name2 in [('Базовый остаточный', 'Остаточный (бутылочное горлышко)'),
                         ('Остаточный (бутылочное горлышко)', 'Широкий остаточный')]:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(histories_res[name1]['test_accs'], label=name1, marker='o')
        plt.plot(histories_res[name2]['test_accs'], label=name2, marker='s')
        plt.title('Сравнение точности на тестовой выборке')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(histories_res[name1]['test_losses'], label=name1, marker='o')
        plt.plot(histories_res[name2]['test_losses'], label=name2, marker='s')
        plt.title('Сравнение потерь на тестовой выборке')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/3_blocks.png")
        plt.show()

if __name__ == "__main__":
    main()