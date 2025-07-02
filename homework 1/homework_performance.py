import torch
import time


def measure_time(operation, device, *args):
    """Измеряет время выполнения операции на указанном устройстве."""
    start_time = time.time()
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    operation(*args)

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
    else:
        elapsed_time = (time.time() - start_time) * 1000

    return elapsed_time


def main():
    if not torch.cuda.is_available():
        # Тест не запускается без CUDA, т.к. это бессмысленно
        print("CUDA недоступно")
        return

    # 3.1 Подготовка данных
    sizes = [
        (64, 1024, 1024),
        (128, 512, 512),
        (256, 256, 256)
    ]

    matrices = []
    for size in sizes:
        cpu_tensor = torch.randn(size)
        gpu_tensor = cpu_tensor.to("cuda")
        matrices.append((cpu_tensor, gpu_tensor))

    # 3.3 Сравнение операций
    operations = [
        ("Матричное умножение", lambda a : a @ a),
        ("Поэлементное сложение", lambda a : a + a),
        ("Поэлементное умножение", lambda a : a * a),
        ("Транспонирование", lambda a: a.mT),
        ("Вычисление суммы", lambda a: torch.sum(a))
    ]

    print("Операция               | CPU (мс) | GPU (мс) | Ускорение")
    print("-" * 56)

    for op_name, op_func in operations:
        cpu_time_total = 0
        gpu_time_total = 0

        for cpu_tensor, gpu_tensor in matrices:
            # Измерение CPU
            cpu_time_total += measure_time(op_func, "cpu", cpu_tensor)
            # Измерение GPU
            gpu_time_total += measure_time(op_func, "cuda", gpu_tensor)

        # Усреднение по всем матрицам
        avg_cpu_time = cpu_time_total / len(matrices)
        avg_gpu_time = gpu_time_total / len(matrices)
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0

        print(f"{op_name:<22} | {avg_cpu_time:>8.1f} | {avg_gpu_time:>8.1f} | {speedup:>7.1f}x")

    # 3.4 Анализ результатов в readme


if __name__ == "__main__":
    main()