import torch
from torch.utils.data import Dataset
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score, \
    confusion_matrix
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from homework_model_modification import LinearRegression, LogisticRegression


# 2.1 Кастомный Dataset класс
class CSVDataset(Dataset):
    """
    Класс для кастомного датасета, который загружает и обрабатывает данные из CSV файла.

    :param path: Путь к CSV файлу
    :param target_column: Название столбца с целевой переменной
    :param task_type: Тип задачи ('regression' или 'classification')
    """

    def __init__(self, path, target_column, task_type='regression'):
        self.task_type = task_type

        # Чтение CSV файла
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data = [row for row in reader]

        # Преобразуем данные в numpy массив
        data = np.array(data)
        col_indices = {col: idx for idx, col in enumerate(header)}

        # Выделяем целевую переменную
        y = data[:, col_indices[target_column]]
        X = np.delete(data, col_indices[target_column], axis=1)

        # Кодируем категориальные признаки
        for i in range(X.shape[1]):
            try:
                X[:, i] = X[:, i].astype(float)
            except ValueError:
                le = LabelEncoder()
                X[:, i] = le.fit_transform(X[:, i])

        X = X.astype(np.float32)

        # Нормализация числовых данных
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Преобразуем целевую переменную в нужный формат
        if task_type == 'classification':
            y = LabelEncoder().fit_transform(y)
            y = y.astype(np.int64)
        else:
            y = y.astype(np.float32)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Визуализация матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Визуализация матрицы ошибок.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix2.png")
    plt.show()


# Основной блок
if __name__ == '__main__':
    # Линейная регрессия
    dataset_reg = CSVDataset('data/regression_dataset.csv', target_column='target', task_type='regression')
    X_train, X_val, y_train, y_val = train_test_split(dataset_reg.X, dataset_reg.y, test_size=0.25)

    model_reg = LinearRegression(in_features=X_train.shape[1])
    criterion_reg = nn.MSELoss()
    optimizer_reg = optim.SGD(model_reg.parameters(), lr=0.01)

    # Обучение модели линейной регрессии
    for epoch in range(100):
        model_reg.train()
        optimizer_reg.zero_grad()
        y_pred = model_reg(X_train)
        loss = criterion_reg(y_pred.squeeze(), y_train)
        loss.backward()
        optimizer_reg.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Оценка модели на валидации
    with torch.no_grad():
        model_reg.eval()
        y_pred_val = model_reg(X_val).squeeze()
        print("MSE (regression):", mean_squared_error(y_val.numpy(), y_pred_val.numpy()))

    # Логистическая регрессия
    dataset_cls = CSVDataset('data/classification_dataset.csv', target_column='label', task_type='classification')
    X_train, X_val, y_train, y_val = train_test_split(dataset_cls.X, dataset_cls.y, test_size=0.25)

    model_cls = LogisticRegression(in_features=X_train.shape[1], num_classes=2)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = optim.SGD(model_cls.parameters(), lr=0.01)

    # Обучение модели логистической регрессии
    for epoch in range(100):
        model_cls.train()
        optimizer_cls.zero_grad()
        y_pred = model_cls(X_train)
        loss = criterion_cls(y_pred, y_train.long())
        loss.backward()
        optimizer_cls.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Оценка модели на валидации
    with torch.no_grad():
        model_cls.eval()
        y_pred_cls = torch.argmax(model_cls(X_val), dim=1)
        print("Accuracy (classification):", accuracy_score(y_val.numpy(), y_pred_cls.numpy()))
        print("Precision:", precision_score(y_val.numpy(), y_pred_cls.numpy()))
        print("Recall:   ", recall_score(y_val.numpy(), y_pred_cls.numpy()))
        print("F1-score: ", f1_score(y_val.numpy(), y_pred_cls.numpy()))

        y_proba = F.softmax(model_cls(X_val), dim=1).numpy()
        y_true_onehot = F.one_hot(y_val.long(), num_classes=2).numpy()
        try:
            auc = roc_auc_score(y_true_onehot, y_proba, multi_class='ovr')
            print("ROC-AUC:  ", auc)
        except:
            print("ROC-AUC не рассчитан.")

        # Визуализация матрицы ошибок
        plot_confusion_matrix(y_val.numpy(), y_pred_cls.numpy(), labels=["Class 0", "Class 1"])
