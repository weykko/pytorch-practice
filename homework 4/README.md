# Домашнее задание к уроку 4: Сверточные сети

## Задание 1: Сравнение CNN и полносвязных сетей

### 1.1 Сравнение на MNIST

Результаты обучения на MNIST:

```
FullyConnected
Train Loss: 0.0829, Train Acc: 0.9752
Test Loss: 0.0864, Test Acc: 0.9767
Параметры: 567434
Время обучения: 160.52 с
Время инференса: 0.000359 с

SimpleCNN
Train Loss: 0.0249, Train Acc: 0.9923
Test Loss: 0.0279, Test Acc: 0.9912
Параметры: 421642
Время обучения: 185.78 с
Время инференса: 0.000334 с

ResidualCNN
Train Loss: 0.0217, Train Acc: 0.9934
Test Loss: 0.0274, Test Acc: 0.9912
Параметры: 160906
Время обучения: 223.07 с
Время инференса: 0.000413 с
```

Сравнение моделей на графиках:

![compare_fcn_cnn_mnist.png](plots%2Fcompare_fcn_cnn_mnist.png)
![compare_cnn_residual_cnn_mnist.png](plots%2Fcompare_cnn_residual_cnn_mnist.png)

Простая CNN и ResidualCNN показали одинаковую точность на тестовой выборке (99.12%), но у SimpleCNN более короткое время обучения и инференса.  
Полносвязная сеть сильно уступает по точности и имеет наибольшее количество параметров.  
Также стоит отметить, что у всех 3 моделей происходит небольшое переобучение на последних эпохах.

### 1.2 Сравнение на CIFAR-10

Результаты обучения на CIFAR-10:

```
FullyConnected
Train Loss: 1.5135, Train Acc: 0.4692
Test Loss: 1.4480, Test Acc: 0.4942
Параметры: 1738890
Время обучения: 144.44 с
Время инференса: 0.000374 с

ResidualCNN
Train Loss: 0.5047, Train Acc: 0.8238
Test Loss: 0.6139, Test Acc: 0.7861
Параметры: 161482
Время обучения: 210.85 с
Время инференса: 0.000406 с

RegularizedResidualCNN
Train Loss: 0.6356, Train Acc: 0.7787
Test Loss: 0.6370, Test Acc: 0.7789
Параметры: 161482
Время обучения: 220.23 с
Время инференса: 0.000412 с
```

Сравнение моделей на графиках:

![compare_fcn_rcnn_cifar.png](plots%2Fcompare_fcn_rcnn_cifar.png)
![compare_rcnn_regularcnn_cifar.png](plots%2Fcompare_rcnn_regularcnn_cifar.png)

Confusion matrix и gradient flow для ResidualCNN:

![ResidualCNN_matrix.png](plots%2FResidualCNN_matrix.png)
![ResidualCNN_gradient.png](plots%2FResidualCNN_gradient.png)

ResidualCNN показала лучшие результаты с точностью 78.61% на тесте, обогнав на последней эпохе модель с регуляризацией. Полносвязная сеть продемонстрировала худшие результаты, что делает её малоприменимой для этой задачи.  
Время обучения было наименьшим у Полносвязной сети, в то время как ResidualCNN и RegularizedResidualCNN потребовали больше времени из-за своей сложности.  
Все модели не показали признаков переобучения.

Остальные результаты по обоим сравнениям в [cnn_architecture_analysis.txt](results%2Fcnn_architecture_analysis.txt) и графики в [plots](plots).