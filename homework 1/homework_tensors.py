import torch


def main():
    # 1.1 Создание тензоров
    # Тензор 3x4 со случайными числами [0, 1)
    tensor_rand = torch.rand(3, 4)

    # Тензор 2x3x4 с нулями
    tensor_zeros = torch.zeros(2, 3, 4)

    # Тензор 5x5 с единицами
    tensor_ones = torch.ones(5, 5)

    # Тензор 4x4 с числами 0-15
    tensor_range = torch.arange(0, 16).reshape(4, 4)

    print("1.1 Создание тензоров:")
    print("Рандомный тензор 3x4:\n", tensor_rand)
    print("Нулевой тензор 2x3x4:\n", tensor_zeros)
    print("Единичный тензор 5x5:\n", tensor_ones)
    print("Рендж тензор 4x4:\n", tensor_range)

    # 1.2 Операции с тензорами
    # Создаем два тензора 3x4 и 4x3
    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    # Проводим операции
    A_transposed = A.T
    matmul_result = A @ B
    elementwise_mul = A * B.T
    sum_A = torch.sum(A)

    print("\n1.2 Операции с тензорами:")
    print("A транспонированный:\n", A_transposed)
    print("Умножение матриц (A * B):\n", matmul_result)
    print("Поэлементное умножение (A * B.T):\n", elementwise_mul)
    print("Сумма элементов A:", sum_A)

    # 1.3 Индексация и срезы
    # Создаем тензор размером 5x5x5
    tensor_5x5x5 = torch.rand(5, 5, 5)

    # Создаем срезы
    first_row = tensor_5x5x5[0, 0, :]
    last_column = tensor_5x5x5[:, :, -1]
    center_matrix = tensor_5x5x5[2:4, 2:4, 2]
    even_indices = tensor_5x5x5[::2, ::2, ::2]

    print("\n1.3 Индексация и срезы:")
    print("Первая строка:\n", first_row)
    print("Последний столбец:\n", last_column)
    print("Подматрица из центра 2x2:\n", center_matrix)
    print("Все элементы с четными индексами:\n", even_indices)

    # 1.4 Работа с формами
    # Создаем тензор размером 24
    tensor_24 = torch.arange(24)

    # Массив форм
    shapes = [
        [2, 12],
        [3, 8],
        [4, 6],
        [2, 3, 4],
        [2, 2, 2, 3]
    ]

    # Преобразуем его в заданные формы
    print("\n1.3 Работа с формами:")
    reshaped_tensors = []
    for shape in shapes:
        reshaped = tensor_24.reshape(shape)
        reshaped_tensors.append(reshaped)
        print(f"\nФорма {"x".join(map(str, shape))}:\n", reshaped)


if __name__ == "__main__":
    main()