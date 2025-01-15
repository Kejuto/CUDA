#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

__global__ void matrixMulKernel(int* C, const int* A, const int* B, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int result = 0;
        for (int k = 0; k < width; ++k) {
            result += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = result;
    }
}


void matrixMultiply(const int* A, const int* B, int* C, int width) {
    int* d_A, * d_B, * d_C;

    // Выделение памяти на устройстве
    cudaMalloc((void**)&d_A, width * width * sizeof(int));
    cudaMalloc((void**)&d_B, width * width * sizeof(int));
    cudaMalloc((void**)&d_C, width * width * sizeof(int));

    // Копирование данных на устройство
    cudaMemcpy(d_A, A, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, width * width * sizeof(int), cudaMemcpyHostToDevice);

    // Определение размеров блоков и сетки
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Запуск ядра
    matrixMulKernel << <numBlocks, threadsPerBlock >> > (d_C, d_A, d_B, width);

    // Копирование результата обратно на хост
    cudaMemcpy(C, d_C, width * width * sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение памяти на устройстве
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    for (int N = 100; N <= 2000; N += 100) {
        int* A = (int*)malloc(N * N * sizeof(int));
        int* B = (int*)malloc(N * N * sizeof(int));
        int* C = (int*)malloc(N * N * sizeof(int));

        // Инициализация матриц A и B случайными значениями
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = rand() % 10; // случайные значения от 0 до 9
                B[i * N + j] = rand() % 10; // случайные значения от 0 до 9
            }
        }

        // Умножение матриц
        auto start = std::chrono::high_resolution_clock::now();
        matrixMultiply(A, B, C, N);
        auto end = std::chrono::high_resolution_clock::now();

        // Проверка корректности результата
        std::chrono::duration<double> elapsed = end - start;

       // Вывод результатов
        printf("Размер матриц: %dx%d\n", N, N);
        printf("Время выполнения: %f секунд\n", elapsed.count());


        // Освобождение памяти
        free(A);
        free(B);
        free(C);
    }

    return 0;
}
