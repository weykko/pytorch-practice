import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
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


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=28 * 28, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class RegularizedCNNWithResidual(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.dropout(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNNKernelSize(nn.Module):
    """
    Модель для анализа влияния размера ядра свертки на производительность.
    """

    def __init__(self, kernel_config, input_channels=3, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = input_channels
        out_channels = [32, 64, 128]  # Фиксированное увеличение количества каналов

        for i, kernel_size in enumerate(kernel_config):
            if isinstance(kernel_size, tuple):  # Для комбинации 1x1 + 3x3
                self.layers.append(nn.Conv2d(in_channels, out_channels[i] // 2, 1))
                self.layers.append(nn.BatchNorm2d(out_channels[i] // 2))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Conv2d(out_channels[i] // 2, out_channels[i], 3, padding=1))
            else:
                self.layers.append(nn.Conv2d(in_channels, out_channels[i], kernel_size, padding=kernel_size // 2))
            self.layers.append(nn.BatchNorm2d(out_channels[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels[i]

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_first_layer_activations(self, x):
        """
        Получаем активации после первого сверточного блока.
        """
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        return x


class CNNDepth(nn.Module):
    """
    Модель для анализа влияния глубины сети на производительность.
    """

    def __init__(self, num_conv_layers, use_residual=False, input_channels=3, num_classes=10):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        in_channels = input_channels
        out_channels = [32, 64, 128, 256, 256, 256][:num_conv_layers]

        for i in range(num_conv_layers):
            if use_residual and i > 0 and i % 2 == 0:
                self.layers.append(ResidualBlock(in_channels, out_channels[i]))
                in_channels = out_channels[i]
            else:
                self.layers.append(nn.Conv2d(in_channels, out_channels[i], 3, padding=1))
                self.layers.append(nn.BatchNorm2d(out_channels[i]))
                self.layers.append(nn.ReLU())
                if i % 2 == 1:
                    self.layers.append(nn.MaxPool2d(2, 2))
                in_channels = out_channels[i]

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(out_channels[-1] * 4 * 4, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_feature_maps(self, x, layer_idx):
        """
        Получаем карты признаков для заданного слоя.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == layer_idx:
                return x
        return x