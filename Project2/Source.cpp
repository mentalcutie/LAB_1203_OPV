#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    string videoPath = "video.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cout << "Ошибка: Не удалось открыть видеофайл." << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame; // Захватываем кадр за кадром

        if (frame.empty()) {
            cout << "Конец видео." << endl;
            break;
        }

        // Преобразуем кадр в оттенки серого для обнаружения контуров
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Применяем размытие Гаусса для уменьшения шума
        Mat blurred;
        GaussianBlur(gray, blurred, Size(5, 5), 0);

        // Выполняем обнаружение краев
        Mat edges;
        Canny(blurred, edges, 20, 150);

        // Находим контуры
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Обрабатываем каждый контур
        for (size_t i = 0; i < contours.size(); i++) {
            // Аппроксимируем контур в многоугольник
            vector<Point> approx;
            approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.01, true);

            // Проверяем, имеет ли контур 4 вершины
            if (approx.size() == 4) {
                // Вычисляем площадь и периметр
                double area = contourArea(contours[i]);
                double perimeter = arcLength(contours[i], true);
                double ratio = sqrt(area) / (perimeter / 4.0); // Соотношение для квадрата

                // Пороги для квадрата с учетом площади и соотношения
                if (ratio > 0.9 && ratio < 1.1 && area > 5000 && area < 55000) { // Ограничение площади для меньшего квадрата
                    // Получаем прямоугольник
                    Rect rect = boundingRect(contours[i]);

                    // Рисуем зеленый прямоугольник
                    rectangle(frame, rect, Scalar(0, 255, 0), 2);

                    // Добавляем метку "Square"
                    putText(frame, "Square", Point(rect.x, rect.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
                }
            }
        }

        // Отображаем результат
        imshow("Обнаружение квадрата", frame);

        // Прерываем цикл по нажатию клавиши 'q'
        if (waitKey(10) == 'q') {
            break;
        }
    }

    // Освобождаем объект захвата видео и закрываем окна
    cap.release();
    destroyAllWindows();

    return 0;
}