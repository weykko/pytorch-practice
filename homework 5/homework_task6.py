import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from utils.datasets import CustomImageDataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_datasets(train_dir, val_dir, transform):
    """
    Загружает тренировочный и валидационный датасеты.
    """
    train_dataset = CustomImageDataset(train_dir, transform=transform)
    val_dataset = CustomImageDataset(val_dir, transform=transform)
    return train_dataset, val_dataset


def prepare_model(num_classes):
    """
    Загружает предобученную модель ResNet18 и заменяет последний слой на количество классов.
    """
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model


def train_model(model, train_loader, optimizer, loss_fn, num_epochs):
    """
    Обучает модель и возвращает список потерь и точности на тренировочных данных.
    """
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Перевод модели в режим обучения
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # Перенос данных на устройство
            optimizer.zero_grad()              # Обнуление градиентов
            out = model(x)                     # Прямой проход
            loss = loss_fn(out, y)             # Вычисление потерь
            loss.backward()                    # Обратное распространение
            optimizer.step()                   # Шаг оптимизации

            # Подсчет метрик
            running_loss += loss.item()
            _, predicted = torch.max(out, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # Средние значения за эпоху
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    return train_losses, train_accuracies


def validate_model(model, val_loader):
    """
    Проверяет точность модели на валидационном датасете.
    """
    model.eval()  # Перевод модели в режим оценки
    val_correct = 0
    val_total = 0

    with torch.no_grad():  # Отключение вычисления градиентов
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, predicted = torch.max(out, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

    val_accuracy = val_correct / val_total
    return val_accuracy


def plot_metrics(train_losses, train_accuracies, num_epochs, path):
    """
    Строит графики потерь и точности.
    """
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Training Loss', color="coral")
    plt.title('Training Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, marker='o', label='Training Accuracy', color="deepskyblue")
    plt.title('Training Accuracy')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def main():
    """
    Основная функция для выполнения обучения и валидации модели.
    """
    # Параметры
    train_dir = './data/train'
    val_dir = './data/test'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    num_epochs = 5

    # Загрузка датасетов
    train_dataset, val_dataset = load_datasets(train_dir, val_dir, transform)

    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Подготовка модели
    num_classes = len(train_dataset.get_class_names())
    model = prepare_model(num_classes)

    # Определение оптимизатора и функции потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Обучение модели
    train_losses, train_accuracies = train_model(model, train_loader, optimizer, loss_fn, num_epochs)

    # Валидация
    val_accuracy = validate_model(model, val_loader)
    print(f'Точность на валидации: {val_accuracy:.4f}')

    # Визуализация процесса обучения
    plot_metrics(train_losses, train_accuracies, num_epochs, "plots/6_metrics.png")


if __name__ == "__main__":
    main()
