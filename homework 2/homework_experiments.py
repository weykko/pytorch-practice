import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from homework_model_modification import LinearRegression
import matplotlib.pyplot as plt


def run_experiment(learning_rates, batch_sizes, optimizers, input_dim=5, epochs=50):
    """
    Эксперименты с гиперпараметрами: скорость обучения, размер батча, оптимизаторы.

    :param learning_rates: Список различных значений скорости обучения
    :param batch_sizes: Список различных значений размера батча
    :param optimizers: Список оптимизаторов для эксперимента
    :param input_dim: Количество признаков для регрессионной задачи
    :param epochs: Количество эпох для обучения
    :return: Словарь с результатами (гиперпараметры -> MSE)
    """
    results = []

    # Генерация данных
    X, y = generate_data(n_samples=500, n_features=input_dim)

    # Разделение на тренировочные и валидационные данные
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

    for lr in learning_rates:
        for bs in batch_sizes:
            for opt_name in optimizers:
                model = LinearRegression(input_dim)
                optimizer = get_optimizer(model, lr, opt_name)

                dataset = TensorDataset(X_train, y_train)
                loader = DataLoader(dataset, batch_size=bs, shuffle=True)

                # Обучение модели
                train_model(model, loader, optimizer, epochs)

                # Оценка на валидации
                mse = evaluate_model(model, X_val, y_val)
                results.append((lr, bs, opt_name, mse))
                print(f"LR: {lr}, BS: {bs}, OPT: {opt_name} -> MSE: {mse:.4f}")

    visualize_results(results)

    return results


def generate_data(n_samples, n_features, noise=10):
    """
    Генерация данных для задачи регрессии.

    :param n_samples: Количество образцов
    :param n_features: Количество признаков
    :param noise: Уровень шума в данных
    :return: Признаки и целевая переменная
    """
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = y.reshape(-1, 1)
    y = StandardScaler().fit_transform(y)
    return X, y


def get_optimizer(model, lr, opt_name):
    """
    Получение оптимизатора в зависимости от его типа.

    :param model: Модель
    :param lr: Скорость обучения
    :param opt_name: Название оптимизатора
    :return: Оптимизатор
    """
    if opt_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)


def train_model(model, dataloader, optimizer, epochs):
    """
    Обучение модели.

    :param model: Модель для обучения
    :param dataloader: Даталоадер с тренировочными данными
    :param optimizer: Оптимизатор
    :param epochs: Количество эпох
    """
    model.train()
    for epoch in range(epochs):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            pred = model(xb)
            pred = pred.squeeze()
            yb = yb.squeeze()
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()


def evaluate_model(model, X_val, y_val):
    """
    Оценка модели на валидационных данных.

    :param model: Обученная модель
    :param X_val: Валидационные данные
    :param y_val: Истинные метки
    :return: MSE
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).squeeze()
        mse = mean_squared_error(y_val.numpy(), y_pred.numpy())
    return mse


def visualize_results(results):
    """
    Визуализация результатов гиперпараметрических экспериментов в виде графика.

    :param results: Список с результатами (гиперпараметры -> MSE)
    """
    # Разделяем результаты по оптимизаторам для отображения разных кривых
    optimizers = set([res[2] for res in results])
    for opt_name in optimizers:
        opt_results = [res for res in results if res[2] == opt_name]
        learning_rates = [res[0] for res in opt_results]
        batch_sizes = [res[1] for res in opt_results]
        mse_values = [res[3] for res in opt_results]

        # Визуализация с использованием matplotlib
        plt.plot(learning_rates, mse_values, marker='o', label=f'{opt_name}')

    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('MSE')
    plt.title('Effect of Learning Rate on MSE for Different Optimizers')
    plt.legend()
    plt.show()
    plt.savefig("plots/hyperparameter_experiments.png")


def feature_engineering():
    """
    Эксперимент с Feature Engineering: добавление полиномиальных признаков для улучшения модели.
    """
    X, y = make_regression(n_samples=300, n_features=3, noise=15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

    # Базовая модель (без полиномиальных признаков)
    model_base = LinearRegression(X_train.shape[1])
    optimizer_base = torch.optim.Adam(model_base.parameters(), lr=0.01)

    # Обучение базовой модели
    train_model(model_base, DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                     torch.tensor(y_train, dtype=torch.float32)),
                                       batch_size=32, shuffle=True), optimizer_base, epochs=100)

    # Оценка базовой модели
    mse_base = evaluate_model(model_base, torch.tensor(X_val, dtype=torch.float32),
                              torch.tensor(y_val, dtype=torch.float32))
    print("MSE for Base Model:", mse_base)

    # Полиномиальные признаки
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Масштабирование признаков
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train_poly = scaler_X.fit_transform(X_train_poly)
    X_val_poly = scaler_X.transform(X_val_poly)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))

    # Преобразуем данные в тензоры
    X_train_poly = torch.tensor(X_train_poly, dtype=torch.float32)
    X_val_poly = torch.tensor(X_val_poly, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Обучение модели с полиномиальными признаками
    model_poly = LinearRegression(X_train_poly.shape[1])
    optimizer_poly = torch.optim.Adam(model_poly.parameters(), lr=0.01)

    # Обучение модели с полиномиальными признаками
    train_model(model_poly, DataLoader(TensorDataset(X_train_poly, y_train), batch_size=32, shuffle=True), optimizer_poly, epochs=100)

    # Оценка модели с полиномиальными признаками
    mse_poly = evaluate_model(model_poly, X_val_poly, y_val)
    print("MSE with Poly Features:", mse_poly)


if __name__ == '__main__':
    # # Эксперименты с гиперпараметрами
    print("Исследование гиперпараметров")
    results = run_experiment(
        learning_rates=[0.001, 0.01, 0.1],
        batch_sizes=[16, 32],
        optimizers=['SGD', 'Adam', 'RMSprop']
    )

    # Feature Engineering
    print("\nFeature Engineering")
    feature_engineering()
