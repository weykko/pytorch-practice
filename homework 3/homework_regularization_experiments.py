import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Устанавливаем устройство (GPU, если доступно, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mnist_loaders(batch_size):
    """
    Функция для загрузки данных MNIST.

    Аргументы:
        batch_size (int): Размер батча.

    Возвращает:
        train_loader (DataLoader): Загрузчик данных для обучения.
        test_loader (DataLoader): Загрузчик данных для тестирования.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class FullyConnectedModel(nn.Module):
    """
    Модель полносвязной нейронной сети для классификации изображений MNIST.
    """

    def __init__(self, input_size, num_classes, layers):
        """
        Инициализация модели.

        Аргументы:
            input_size (int): Размер входного слоя.
            num_classes (int): Количество классов для классификации.
            layers (list): Конфигурация скрытых слоев.
        """
        super(FullyConnectedModel, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for layer in layers:
            if layer["type"] == "linear":
                self.layers.append(nn.Linear(prev_size, layer["size"]))
                prev_size = layer["size"]
            elif layer["type"] == "relu":
                self.layers.append(nn.ReLU())
            elif layer["type"] == "dropout":
                self.layers.append(nn.Dropout(p=layer["rate"]))
            elif layer["type"] == "batch_norm":
                self.layers.append(nn.BatchNorm1d(prev_size))
        self.layers.append(nn.Linear(prev_size, num_classes))

    def forward(self, x):
        """
        Прямой проход через модель.

        Аргументы:
            x (tensor): Входной тензор.

        Возвращает:
            tensor: Выход модели.
        """
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


def train_model(model, train_loader, test_loader, epochs, device, optimizer):
    """
    Функция для обучения модели.

    Аргументы:
        model (nn.Module): Модель для обучения.
        train_loader (DataLoader): Загрузчик обучающих данных.
        test_loader (DataLoader): Загрузчик тестовых данных.
        epochs (int): Количество эпох для обучения.
        device (torch.device): Устройство для вычислений (GPU или CPU).
        optimizer (torch.optim.Optimizer): Оптимизатор для обновления весов модели.

    Возвращает:
        dict: История потерь и точности на обучении и тестировании.
    """
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = 100. * correct / total
        history['test_acc'].append(test_acc)
        print(
            f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    return history


def plot_training_history(history, title):
    """
    Построение графиков потерь и точности.

    Аргументы:
        history (dict): История потерь и точности.
        title (str): Заголовок для графиков.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Потери на обучении')
    plt.title(f'{title} - Потери')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Точность на обучении')
    plt.plot(history['test_acc'], label='Точность на тесте')
    plt.title(f'{title} - Точность')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{title}.png")
    plt.show()


# Функции для добавления регуляризации
def add_dropout(layers, rate):
    """
    Функция для добавления Dropout после каждого слоя ReLU.

    Аргументы:
        layers (list): Список слоев.
        rate (float): Коэффициент Dropout.

    Возвращает:
        list: Новый список слоев с добавлением Dropout.
    """
    new_layers = []
    for layer in layers:
        new_layers.append(layer)
        if layer["type"] == "relu":
            new_layers.append({"type": "dropout", "rate": rate})
    return new_layers


def add_batchnorm(layers):
    """
    Функция для добавления BatchNorm после каждого линейного слоя.

    Аргументы:
        layers (list): Список слоев.

    Возвращает:
        list: Новый список слоев с добавлением BatchNorm.
    """
    new_layers = []
    for layer in layers:
        if layer["type"] == "linear":
            new_layers.append(layer)
            new_layers.append({"type": "batch_norm"})
        else:
            new_layers.append(layer)
    return new_layers


# 3.1 Сравнение методов регуляризации
def compare_regularization_methods(train_loader, test_loader):
    """
    Эксперимент 3.1: Сравнение методов регуляризации.

    Создает и обучает модели с различными методами регуляризации, затем выводит результаты.

    Аргументы:
        train_loader (DataLoader): Загрузчик обучающих данных.
        test_loader (DataLoader): Загрузчик тестовых данных.
    """
    base_layers = [
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"}
    ]

    models = {
        "Без регуляризации": FullyConnectedModel(784, 10, base_layers).to(device),
        "Dropout 0.1": FullyConnectedModel(784, 10, add_dropout(base_layers, 0.1)).to(device),
        "Dropout 0.3": FullyConnectedModel(784, 10, add_dropout(base_layers, 0.3)).to(device),
        "Dropout 0.5": FullyConnectedModel(784, 10, add_dropout(base_layers, 0.5)).to(device),
        "BatchNorm": FullyConnectedModel(784, 10, add_batchnorm(base_layers)).to(device),
        "Dropout 0.3 + BatchNorm": FullyConnectedModel(784, 10, add_dropout(add_batchnorm(base_layers), 0.3)).to(
            device),
        "L2 Regularization": FullyConnectedModel(784, 10, base_layers).to(device)
    }

    histories = {}
    for name, model in models.items():
        print(f"\nОбучение {name}")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001 if name == "L2 Regularization" else 0.0)
        history = train_model(model, train_loader, test_loader, epochs=10, device=device, optimizer=optimizer)
        histories[name] = history
        plot_training_history(history, name)

    # Визуализация распределения весов
    plot_weight_distribution(models)


def plot_weight_distribution(models):
    """
    Строит гистограммы распределения весов

    Аргументы:
        models (dict): Словарь моделей для анализа весов.
    """
    plt.figure(figsize=(12, 8))
    for name, model in models.items():
        weights = torch.cat([p.view(-1) for p in model.parameters()]).cpu().detach().numpy()
        sns.histplot(weights, label=name, bins=50, alpha=0.5)
    plt.legend()
    plt.title('Распределения весов')
    plt.tight_layout()
    plt.savefig("plots/weight_distribution.png")
    plt.show()


def adaptive_regularization(train_loader, test_loader):
    """
    Эксперимент 3.2: Адаптивная регуляризация.

    Создает и обучает модели с адаптивным Dropout и BatchNorm с разными значениями momentum.

    Аргументы:
        train_loader (DataLoader): Загрузчик обучающих данных.
        test_loader (DataLoader): Загрузчик тестовых данных.
    """

    # Адаптивный Dropout
    class ScheduledDropout(nn.Module):
        def __init__(self, initial_rate, final_rate, total_epochs):
            super().__init__()
            self.initial_rate = initial_rate
            self.final_rate = final_rate
            self.total_epochs = total_epochs
            self.current_epoch = 0

        def forward(self, x):
            rate = self.initial_rate + (self.final_rate - self.initial_rate) * (self.current_epoch / self.total_epochs)
            return nn.functional.dropout(x, p=rate, training=self.training)

        def step(self):
            self.current_epoch += 1

    class AdaptiveModel(nn.Module):
        def __init__(self, input_size, num_classes, use_adaptive_dropout=False, momentum=0.1):
            super(AdaptiveModel, self).__init__()
            self.layers = nn.ModuleList()
            prev_size = input_size
            for i in range(3):
                self.layers.append(nn.Linear(prev_size, 256))
                self.layers.append(nn.BatchNorm1d(256, momentum=momentum))
                if use_adaptive_dropout:
                    self.layers.append(ScheduledDropout(0.5, 0.1, 10))
                else:
                    self.layers.append(nn.Dropout(0.3))
                self.layers.append(nn.ReLU())
                prev_size = 256
            self.layers.append(nn.Linear(prev_size, num_classes))

        def forward(self, x):
            x = x.view(x.size(0), -1)
            for layer in self.layers:
                x = layer(x)
            return x

        def step_dropout(self):
            for layer in self.layers:
                if isinstance(layer, ScheduledDropout):
                    layer.step()

    adaptive_models = {
        "Адаптивный Dropout": AdaptiveModel(784, 10, use_adaptive_dropout=True).to(device),
        "BatchNorm Momentum 0.1": AdaptiveModel(784, 10, momentum=0.1).to(device),
        "BatchNorm Momentum 0.9": AdaptiveModel(784, 10, momentum=0.9).to(device),
        "Комбинированный": AdaptiveModel(784, 10, use_adaptive_dropout=True, momentum=0.5).to(device)
    }

    adaptive_histories = {}
    for name, model in adaptive_models.items():
        print(f"\nОбучение {name}...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        history = train_model(model, train_loader, test_loader, epochs=10, device=device, optimizer=optimizer)
        if "Адаптивный Dropout" in name:
            model.step_dropout()
        adaptive_histories[name] = history
        plot_training_history(history, name)

    # Анализ весов слоев для комбинированной модели
    plot_weight_distribution_adaptive(adaptive_models)


def plot_weight_distribution_adaptive(adaptive_models):
    """
    Строит гистограммы распределения весов для комбинированной модели.

    Аргументы:
        models (dict): Словарь моделей для анализа весов.
    """
    combined_model = adaptive_models["Комбинированный"]
    layer_weights = [layer.weight.view(-1).cpu().detach().numpy() for layer in combined_model.layers if
                     isinstance(layer, nn.Linear)]

    plt.figure(figsize=(12, 8))
    for i, weights in enumerate(layer_weights):
        sns.histplot(weights, label=f'Слой {i + 1}', bins=50, alpha=0.5)
    plt.legend()
    plt.title('Распределения весов по слоям - Комбинированная модель')
    plt.tight_layout()
    plt.savefig("plots/weight_distribution_adaptive.png")
    plt.show()


def main():
    # Загружаем набор данных MNIST
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    # 3.1 Сравнение техник регуляризации
    print("3.1: Сравнение методов регуляризации")
    compare_regularization_methods(train_loader, test_loader)

    # 3.2 Адаптивная регуляризация
    print("\n3.2: Адаптивная регуляризация")
    adaptive_regularization(train_loader, test_loader)


if __name__ == '__main__':
    main()
