Функции для перемножения матриц:
  cpu(matrix_a, matrix_b): выполняет матричное умножение на CPU с использованием NumPy.
  gpu(matrix_a, matrix_b): выполняет матричное умножение на GPU с использованием CuPy.
Основная функция main():
  Создает матрицы различных размеров.
  Вызывает функции для перемножения на CPU и GPU.
  Сравнивает результаты и измеряет время выполнения.
Что распараллелено:
  Каждое умножение элементов матриц может выполняться параллельно.
  GPU может обрабатывать тысячи операций одновременно, что позволяет значительно ускорить вычисления для больших матриц по сравнению с CPU.
