#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;
using namespace chrono;

const int IMAGE_DIMENSION = 729;  // Размер изображения

// Функция для рисования фрактала
void generateFractal(Mat& image, int x, int y, int size, int level) {
    if (level <= 0) return;

    int subSize = size / 3;  // Размер меньших квадратов

    // Закрашиваем центральный квадрат
    rectangle(image, Point(x + subSize, y + subSize), Point(x + 2 * subSize, y + 2 * subSize), Scalar(255, 255, 255), FILLED);

    // Параллельный вызов для каждого из 8 подрекурсивных квадратов
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i != 1 || j != 1) {  // Пропускаем центральный квадрат
                generateFractal(image, x + i * subSize, y + j * subSize, subSize, level - 1);
            }
        }
    }
}

int main() {
    int threadsCount = 4;  // Количество потоков для OpenMP
    omp_set_num_threads(threadsCount);

    int recursionDepth;
    cout << "Введите глубину рекурсии: ";
    cin >> recursionDepth;

    // Параллельное выполнение блока вывода
#pragma omp parallel
    {
#pragma omp single
        std::cout << "Используемое количество потоков: " << omp_get_num_threads() << std::endl;
    }

    // Создание черного изображения
    Mat img(IMAGE_DIMENSION, IMAGE_DIMENSION, CV_8UC3, Scalar(0, 0, 0));

    // Засекаем время построения фрактала
    auto startTime = high_resolution_clock::now();

    // Рисуем фрактал
    generateFractal(img, 0, 0, IMAGE_DIMENSION, recursionDepth);

    // Засекаем время завершения
    auto endTime = high_resolution_clock::now();
    auto elapsedTime = duration_cast<milliseconds>(endTime - startTime);

    // Выводим время построения
    cout << "Время создания фрактала: " << elapsedTime.count() << " мс" << endl;
    cout << "----------------------------------" << endl;

    // Сохраняем изображение в текущей директории
    string filename = "sierpinski_fractal.png";
    if (imwrite(filename, img)) {
        cout << "Изображение успешно сохранено как " << filename << endl;
    }
    else {
        cout << "Ошибка сохранения изображения!" << endl;
    }

    // Отображаем изображение
    imshow("Sierpinski Fractal", img);
    waitKey(0);

    return 0;
}
