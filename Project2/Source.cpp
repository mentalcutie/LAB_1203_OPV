#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>

int main() {
    // Шаг 1: Загрузка изображений
    std::vector<cv::Mat> images;
    images.push_back(cv::imread("image1.jpg"));
    images.push_back(cv::imread("image2.jpg"));

    // Проверка на успешную загрузку изображений
    if (images[0].empty() || images[1].empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображения." << std::endl;
        return -1;
    }

    // Шаг 2: Обнаружение ключевых точек и вычисление дескрипторов
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector->detect(images[0], keypoints1);
    detector->detect(images[1], keypoints2);
    extractor->compute(images[0], keypoints1, descriptors1);
    extractor->compute(images[1], keypoints2, descriptors2);

    // Шаг 3: Сопоставление дескрипторов
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> matches_knn;
    matcher.knnMatch(descriptors1, descriptors2, matches_knn, 2);

    // Шаг 4: Фильтрация совпадений с помощью Ratio Test
    std::vector<cv::DMatch> good_matches;
    for (const auto& m : matches_knn) {
        if (m[0].distance < 0.7 * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    // Шаг 5: Вычисление матрицы гомографии с использованием RANSAC
    std::vector<cv::Point2f> points1, points2;
    for (const auto& m : good_matches) {
        points1.push_back(keypoints1[m.queryIdx].pt);
        points2.push_back(keypoints2[m.trainIdx].pt);
    }

    cv::Mat homography = cv::findHomography(points2, points1, cv::RANSAC);

    // Шаг 6: Сшивание изображений
    // Определяем размер панорамы
    int width = images[0].cols + images[1].cols;
    int height = std::max(images[0].rows, images[1].rows);
    cv::Mat panorama = cv::Mat::zeros(height, width, images[0].type());

    // Копируем первое изображение в панораму
    images[0].copyTo(panorama(cv::Rect(0, 0, images[0].cols, images[0].rows)));

    // Преобразуем и накладываем второе изображение
    cv::Mat warped;
    cv::warpPerspective(images[1], warped, homography, cv::Size(width, height));
    warped.copyTo(panorama, warped > 0); // Накладываем только непустые пиксели

    // Шаг 7: Обрезка лишних черных областей
    // Создаем маску для поиска непустых пикселей
    cv::Mat mask;
    cv::cvtColor(panorama, mask, cv::COLOR_BGR2GRAY);
    cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

    // Находим контуры
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Определяем ограничивающий прямоугольник для всех контуров
    cv::Rect bounding_rect;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        bounding_rect = bounding_rect | rect;
    }

    // Обрезаем изображение
    cv::Mat cropped_panorama = panorama(bounding_rect);

    // Шаг 8: Сохранение результата
    cv::imwrite("cropped_panorama.jpg", cropped_panorama);

    std::cout << "Панорама успешно сохранена как 'cropped_panorama.jpg'." << std::endl;

    return 0;
}