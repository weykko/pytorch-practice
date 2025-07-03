import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from utils import make_regression_data, log_epoch, RegressionDataset, ClassificationDataset


# 1.1 Расширение линейной регрессии
class LinearRegression(nn.Module):
    """
    Класс для линейной регрессии с L1 и L2 регуляризацией
    """

    def __init__(self, in_features, l1_lambda=0.0, l2_lambda=0.0):
        """
        Инициализация модели линейной регрессии с параметрами регуляризации.

        :param in_features: Количество входных признаков
        :param l1_lambda: Коэффициент L1 регуляризации
        :param l2_lambda: Коэффициент L2 регуляризации
        """
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def l1_regularization(self):
        return torch.sum(torch.abs(self.linear.weight))

    def l2_regularization(self):
        return torch.sum(self.linear.weight ** 2)


def train_linear_regression(model, dataloader, criterion, optimizer, epochs=100, early_stopping_patience=10):
    """
    Обучение линейной регрессии с добавлением регуляризации и ранней остановки.

    :param model: Модель линейной регрессии
    :param dataloader: Даталоадер для загрузки данных
    :param criterion: Функция потерь
    :param optimizer: Оптимизатор
    :param epochs: Количество эпох для обучения
    :param early_stopping_patience: Количество эпох без улучшений для ранней остановки
    :return: Обученная модель
    """
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            # Добавляем регуляризацию
            l1_loss = model.l1_regularization() * model.l1_lambda
            l2_loss = model.l2_regularization() * model.l2_lambda
            total_loss = loss + l1_loss + l2_loss

            total_loss.backward()
            optimizer.step()

            total_loss += total_loss.item()

        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

    return model


# 1.2 Расширение логистической регрессии
class LogisticRegression(nn.Module):
    """
    Класс для многоклассовой логистической регрессии
    """

    def __init__(self, in_features, num_classes):
        """
        Инициализация модели логистической регрессии для многоклассовой классификации.

        :param in_features: Количество входных признаков
        :param num_classes: Количество классов для классификации
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def train_logistic_regression(model, dataloader, criterion, optimizer, epochs=100):
    """
    Обучение логистической регрессии с вычислением метрик precision, recall, F1-score и ROC-AUC.

    :param model: Модель логистической регрессии
    :param dataloader: Даталоадер для загрузки данных
    :param criterion: Функция потерь
    :param optimizer: Оптимизатор
    :param epochs: Количество эпох для обучения
    :return: Обученная модель
    """
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_labels = []
        all_probs = []  # Массив для вероятностей

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_X.float())
            loss = criterion(logits, batch_y.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Получаем предсказания и метрики
            y_pred = torch.argmax(logits, dim=1)
            acc = (y_pred == batch_y).float().mean()
            total_acc += acc.item()

            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

            # Добавляем вероятности в all_probs
            probs = torch.softmax(logits, dim=1)  # Применяем softmax для многоклассовых вероятностей
            all_probs.extend(probs.cpu().detach().numpy())

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        try:
            # Используем вероятности для ROC-AUC
            roc_auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
        except ValueError:
            roc_auc = 0.0  # Если ROC-AUC не может быть рассчитан для одного класса

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    return model



def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Визуализация матрицы ошибок.

    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :param classes: Названия классов
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.show()


if __name__ == '__main__':
    # Генерация данных для линейной регрессии
    X_reg, y_reg = make_regression_data(n=200)
    dataset_reg = RegressionDataset(X_reg, y_reg)
    dataloader_reg = DataLoader(dataset_reg, batch_size=32, shuffle=True)

    # Создание и обучение модели линейной регрессии
    model_reg = LinearRegression(in_features=1, l1_lambda=0.01, l2_lambda=0.01)
    criterion_reg = nn.MSELoss()
    optimizer_reg = optim.SGD(model_reg.parameters(), lr=0.1)
    trained_model_reg = train_linear_regression(model_reg, dataloader_reg, criterion_reg, optimizer_reg, epochs=100)

    torch.save(trained_model_reg.state_dict(), 'models/linreg_model.pth')

    # Генерация данных для логистической регрессии (многоклассовая классификация)
    X_cls, y_cls = make_classification(n_samples=300, n_features=4, n_classes=3,
                                       n_informative=4, n_redundant=0, random_state=42)

    # Преобразование данных в формат PyTorch
    X_cls = torch.tensor(X_cls, dtype=torch.float32)
    y_cls = torch.tensor(y_cls, dtype=torch.long)
    dataset_cls = ClassificationDataset(X_cls, y_cls)
    dataloader_cls = DataLoader(dataset_cls, batch_size=32, shuffle=True)

    # Создание и обучение модели логистической регрессии
    model_cls = LogisticRegression(in_features=4, num_classes=3)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = optim.SGD(model_cls.parameters(), lr=0.1)
    trained_model_cls = train_logistic_regression(model_cls, dataloader_cls, criterion_cls, optimizer_cls, epochs=100)

    torch.save(trained_model_cls.state_dict(), 'models/logreg_model.pth')

    # Визуализация матрицы ошибок для логистической регрессии
    all_preds_cls = []
    all_labels_cls = []
    for batch_X, batch_y in dataloader_cls:
        logits = trained_model_cls(batch_X)
        y_pred_cls = torch.argmax(logits, dim=1)
        all_preds_cls.extend(y_pred_cls.cpu().numpy())
        all_labels_cls.extend(batch_y.cpu().numpy())

    plot_confusion_matrix(all_labels_cls, all_preds_cls, classes=['Class 0', 'Class 1', 'Class 2'])
