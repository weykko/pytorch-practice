import torch


def main():
    # 2.1 Простые вычисления с градиентами
    # Создаем тензоры x, y, z
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)

    # Вычисляем функцию
    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z

    f.backward()

    print("2.1 Градиенты функции f(x,y,z):")
    print(f"df/dx = {x.grad.item()} (Аналитический расчет: {2 * x + 2 * y * z})")
    print(f"df/dy = {y.grad.item()} (Аналитический расчет: {2 * y + 2 * x * z})")
    print(f"df/dz = {z.grad.item()} (Аналитический расчет: {2 * z + 2 * x * y})")

    # 2.2 Градиент функции потерь MSE
    # Пример данных
    x_data = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 4.0, 6.0])
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    # Линейная функция: y_pred = w * x + b
    y_pred = w * x_data + b

    # Вычисление MSE:
    mse = torch.mean((y_pred - y_true) ** 2)

    mse.backward()

    print("\n2.2 Градиенты MSE:")
    print(f"dL/dw = {w.grad.item()}")
    print(f"dL/db = {b.grad.item()}")

    # 2.3 Цепное правило
    # Создаем тензор
    x = torch.tensor(1.0, requires_grad=True)

    # Вычисляем функцию
    f = torch.sin(x ** 2 + 1)

    # Вычисляем градиент, сохраняя граф
    f.backward(retain_graph=True)

    # Проверка через torch.autograd.grad
    grad_autograd = torch.autograd.grad(f, x, retain_graph=True)[0]

    print("\n2.3 Цепное правило:")
    print(f"df/dx (backward): {x.grad.item()}")
    print(f"df/dx (autograd): {grad_autograd.item()}")


if __name__ == "__main__":
    main()
