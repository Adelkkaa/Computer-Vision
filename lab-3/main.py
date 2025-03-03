import cv2
import numpy as np

# Глобальные переменные для водоразделов
drawing = False  # Флаг для отслеживания, рисуем ли мы
markers = None  # Матрица маркеров
marker_image = None  # Изображение с маркерами
current_marker = 1  # Текущий маркер (1 для объекта, 2 для фона)

# Функция обработки событий мыши
def mouse_callback(event, x, y, flags, param):
    global drawing, markers, marker_image, current_marker

    if event == cv2.EVENT_LBUTTONDOWN:  # Левый клик — объект
        drawing = True
        current_marker = 1
        cv2.circle(marker_image, (x, y), 5, (255, 255, 255), -1)  # Белый цвет для объекта
        cv2.circle(markers, (x, y), 5, current_marker, -1)

    elif event == cv2.EVENT_RBUTTONDOWN:  # Правый клик — фон
        drawing = True
        current_marker = 2
        cv2.circle(marker_image, (x, y), 5, (0, 0, 255), -1)  # Красный цвет для фона
        cv2.circle(markers, (x, y), 5, current_marker, -1)

    elif event == cv2.EVENT_MOUSEMOVE:  # Движение мыши с зажатой кнопкой
        if drawing:
            if current_marker == 1:  # Рисуем объект
                cv2.circle(marker_image, (x, y), 5, (255, 255, 255), -1)
                cv2.circle(markers, (x, y), 5, current_marker, -1)
            elif current_marker == 2:  # Рисуем фон
                cv2.circle(marker_image, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(markers, (x, y), 5, current_marker, -1)

    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:  # Отпустили кнопку мыши
        drawing = False

# Функция для отображения двух изображений: оригинального и сегментированного
def show_images(original_title, original_image, segmented_title, segmented_image):
    cv2.imshow(original_title, original_image)
    cv2.imshow(segmented_title, segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 1. Сегментация с помощью выделения контуров
def contour_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(sobel_combined)
    ret, edges = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output

# 2. Морфологическая сегментация с использованием водоразделов
def watershed_segmentation(image):
    global markers
    markers = cv2.watershed(image, markers)
    washed = np.zeros_like(image, dtype=np.uint8)
    washed[markers == 1] = [0, 255, 0]  # Объекты (зеленый)
    washed[markers == 2] = [255, 0, 0]  # Фон (синий)
    washed[markers == -1] = [0, 0, 255]  # Границы (красный)
    return washed

# 3. Пороговая сегментация (глобальный порог)
def global_threshold_segmentation(image, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh

# 4. Пороговая сегментация (адаптивный порог)
def adaptive_threshold_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

# 5. Сегментация методом K-средних
def kmeans_segmentation(image, K=2):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    if center[0].mean() < center[1].mean():
        object_color = np.array([255, 0, 0], dtype=np.uint8)
        background_color = np.array([0, 0, 255], dtype=np.uint8)
    else:
        object_color = np.array([0, 0, 255], dtype=np.uint8)
        background_color = np.array([255, 0, 0], dtype=np.uint8)
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    segmented_image[label.reshape(image.shape[:2]) == 0] = object_color
    segmented_image[label.reshape(image.shape[:2]) == 1] = background_color
    return segmented_image

# 6. Сегментация методом поиска минимальных разрезов графа (GrabCut)
def grabcut_segmentation(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)
    print(rect)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    output = image * mask2[:, :, np.newaxis]
    return output

# Основная функция
def segment_image(filename, method, params=None):
    global markers, marker_image

    image = cv2.imread(filename)
    if image is None:
        print("Ошибка: не удалось открыть изображение!")
        return

    if method == 1:
        result = contour_segmentation(image)
        show_images("Original Image", image, "Contour Segmentation", result)
    elif method == 2:
        # Инициализация маркеров для водоразделов
        markers = np.zeros(image.shape[:2], dtype=np.int32)
        marker_image = image.copy()

        # Создание окна и привязка обработчика событий мыши
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_callback)

        print("Левый клик — объект, правый клик — фон. Нажмите 'q' для сегментации.")

        while True:
            cv2.imshow("Image", marker_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Нажатие 'q' для запуска сегментации
                break

        result = watershed_segmentation(image)
        show_images("Original Image", image, "Watershed Segmentation", result)
    elif method == 3:
        if params is None:
            params = [127]
        result = global_threshold_segmentation(image, threshold=params[0])
        show_images("Original Image", image, "Global Threshold Segmentation", result)
    elif method == 4:
        result = adaptive_threshold_segmentation(image)
        show_images("Original Image", image, "Adaptive Threshold Segmentation", result)
    elif method == 5:
        if params is None:
            params = [2]
        result = kmeans_segmentation(image, K=params[0])
        show_images("Original Image", image, "K-means Segmentation", result)
    elif method == 6:
        result = grabcut_segmentation(image)
        show_images("Original Image", image, "GrabCut Segmentation", result)
    else:
        print("Неверный номер метода!")

def main():
    filename = "assets/image.jpg"

    while True:
        print("\nМетоды сегментации:")
        print("1. Сегментация по контурам")
        print("2. Морфологическая сегментация (водораздел)")
        print("3. Пороговая сегментация (глобальный порог)")
        print("4. Пороговая сегментация (адаптивный порог)")
        print("5. Сегментация методом K-средних")
        print("6. Сегментация методом поиска минимальных разрезов графа")
        print("0. Выход")

        method = int(input("\nВведите номер метода сегментации (0 для выхода): "))

        if method == 0:
            print("Выход из программы.")
            break

        params = None

        if method == 3:
            threshold = int(input("Введите значение порога (например, 127): "))
            params = [threshold]
        elif method == 5:
            k = int(input("Введите число кластеров (например, 2): "))
            params = [k]

        segment_image(filename, method, params)

if __name__ == "__main__":
    main()