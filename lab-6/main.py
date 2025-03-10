import cv2  # OpenCV для обработки изображений
import numpy as np  # NumPy для работы с массивами и матрицами
import glob  # Для поиска файлов по шаблону
import os  # Для работы с операционной системой


# Функция калибровки камеры (Метод Богета)
def calibrate_camera(chessboard_images_path, chessboard_size=(7, 11)):
    # Создаем массив реальных координат углов шахматной доски
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)  # Заполняем X и Y координаты

    objpoints = []  # Список для хранения реальных координат углов
    imgpoints = []  # Список для хранения найденных углов на изображениях

    # Находим все PNG-изображения в указанной папке
    images = glob.glob(os.path.join(chessboard_images_path, '*.png'))
    for img_path in images:
        img = cv2.imread(img_path, 0)  # Читаем изображение в оттенках серого
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)  # Находим углы шахматной доски
        if ret:  # Если углы найдены
            objpoints.append(objp)  # Добавляем реальные координаты
            imgpoints.append(corners)  # Добавляем найденные углы

    # Проверяем, что были найдены углы хотя бы на одном изображении
    if not objpoints or not imgpoints:
        print("Не удалось найти углы шахматной доски.")
        return None, None

    # Выполняем калибровку камеры
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[::-1], None, None
    )

    return camera_matrix, dist_coeffs  # Возвращаем матрицу камеры и коэффициенты искажения


# Функция стереокалибровки (Метод Богета)
def stereo_calibrate(camera_matrix, dist_coeffs, imgL, imgR):
    sift = cv2.SIFT_create()  # Создаем объект SIFT для детектирования ключевых точек
    kp1, des1 = sift.detectAndCompute(imgL, None)  # Находим ключевые точки и их описатели на левом изображении
    kp2, des2 = sift.detectAndCompute(imgR, None)  # Находим ключевые точки и их описатели на правом изображении

    bf = cv2.BFMatcher()  # Создаем объект BFMatcher для сопоставления описателей
    matches = bf.knnMatch(des1, des2, k=2)  # Находим две лучших соответствия для каждой точки

    # Отфильтровываем плохие соответствия
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Проверяем, что есть достаточно хороших соответствий
    if len(good_matches) < 8:
        print("Недостаточно совпадений.")
        return None, None

    # Преобразуем соответствия в массивы координат
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Вычисляем фундаментальную матрицу
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    # Вычисляем эссенциальную матрицу
    E = camera_matrix.T @ F @ camera_matrix
    # Вычисляем матрицу вращения (R) и вектор смещения (T)
    _, R, T, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)

    return R, T  # Возвращаем матрицу вращения и вектор смещения


# Функция ректификации изображений (Метод Богета)
def rectify_images(imgL, imgR, camera_matrix, dist_coeffs, R, T):
    h, w = imgL.shape  # Получаем размеры изображения
    # Вычисляем новые матрицы проекции и матрицы ректификации
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix, dist_coeffs, camera_matrix, dist_coeffs, (w, h), R, T
    )

    # Создаем карты преобразования для исправления искажений и применения ректификации
    map1L, map2L = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)


    # Применяем карты преобразования к изображениям
    rectified_imgL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
    rectified_imgR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)

    return rectified_imgL, rectified_imgR  # Возвращаем откалиброванные изображения


# Функция вычисления фундаментальной матрицы и стереоректификации (Метод Хартли)
def fundamental_matrix_method(imgL, imgR):
    sift = cv2.SIFT_create()  # Создаем объект SIFT для детектирования ключевых точек
    kp1, des1 = sift.detectAndCompute(imgL, None)  # Находим ключевые точки и их описатели на левом изображении
    kp2, des2 = sift.detectAndCompute(imgR, None)  # Находим ключевые точки и их описатели на правом изображении

    bf = cv2.BFMatcher()  # Создаем объект BFMatcher для сопоставления описателей
    matches = bf.knnMatch(des1, des2, k=2)  # Находим два лучших соответствия для каждой точки

    # Отфильтровываем плохие соответствия, используя правило Лоу (Low's ratio test)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Проверяем, что есть достаточно хороших соответствий
    if len(good_matches) < 8:
        print("Недостаточно совпадений.")
        return None, None

    # Преобразуем соответствия в массивы координат
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Вычисляем фундаментальную матрицу
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

    # Проверяем, что фундаментальная матрица успешно вычислена
    if F is None:
        print("Ошибка: не удалось вычислить фундаментальную матрицу.")
        return None, None

     # Вычисляем гомографии для ректификации с помощью фундаментальной матрицы
    h1, w1 = imgL.shape[:2]
    h2, w2 = imgR.shape[:2]

    _, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, imgSize=(w1, h1))

    # Применяем гомографии к изображениям
    rectified_imgL = cv2.warpPerspective(imgL, H1, (w1, h1))
    rectified_imgR = cv2.warpPerspective(imgR, H2, (w2, h2))

    return rectified_imgL, rectified_imgR  # Возвращаем откалиброванные изображения


# Функция отображения всех изображений в одном окне
def show_results(imgL, imgR, rectifiedL, rectifiedR):
    # Изменяем размер изображений для удобства просмотра
    imgL_resized = cv2.resize(imgL, (400, 400))
    imgR_resized = cv2.resize(imgR, (400, 400))
    rectifiedL_resized = cv2.resize(rectifiedL, (400, 400))
    rectifiedR_resized = cv2.resize(rectifiedR, (400, 400))
    

    # Объединяем изображения горизонтально
    top_row = np.hstack((imgL_resized, imgR_resized))
    bottom_row = np.hstack((rectifiedL_resized, rectifiedR_resized))
    combined = np.vstack((top_row, bottom_row))  # Объединяем строки вертикально

    # Отображаем результаты
    cv2.imshow("Original (Top) | Rectified (Bottom)", combined)
    cv2.waitKey(0)  # Ждем нажатия клавиши
    cv2.destroyAllWindows()  # Закрываем окно

    # Сохраняем результаты
    cv2.imwrite("results/output_image_left.png", rectifiedL)
    cv2.imwrite("results/output_image_right.png", rectifiedR)
    print("Изображения сохранены в папке 'results'.")


# Меню выбора метода
def main():
    # Печатаем меню
    print("\nВыберите метод калибровки стереосистемы камер:")
    print("1 - Калибровка методом Богета (с предварительной калибровкой)")
    print("2 - Калибровка методом Хартли (фундаментальная матрица, без калибровки)")
    print("0 - Выход")

    # Пытаемся получить выбор пользователя
    try:
        choice = int(input("Введите номер метода: "))
    except ValueError:
        print("Ошибка: введите число.")
        return

    # Обработка выхода из программы
    if choice == 0:
        print("Выход из программы.")
        return

    # Загружаем левое и правое изображения
    imgL_path = 'assets/input_image_left.png'
    imgR_path = 'assets/input_image_right.png'
    imgL = cv2.imread(imgL_path, 0)
    imgR = cv2.imread(imgR_path, 0)

    # Проверяем, что изображения успешно загружены
    if imgL is None or imgR is None:
        print("Ошибка: не удалось загрузить изображения.")
        return

    # Обработка выбора метода
    match choice:
        case 1:  # Метод Богета
            chessboard_images_path = 'assets'  # Путь к изображениям шахматной доски
            camera_matrix, dist_coeffs = calibrate_camera(chessboard_images_path)  # Калибруем камеру

            # Проверяем успешность калибровки
            if camera_matrix is None or dist_coeffs is None:
                print("Ошибка калибровки камеры.")
                return

            R, T = stereo_calibrate(camera_matrix, dist_coeffs, imgL, imgR)  # Выполняем стереокалибровку

            # Проверяем успешность стереокалибровки
            if R is None or T is None:
                print("Ошибка стереокалибровки.")
                return

            rectified_imgL, rectified_imgR = rectify_images(imgL, imgR, camera_matrix, dist_coeffs, R, T)  # Ректифицируем изображения
            show_results(imgL, imgR, rectified_imgL, rectified_imgR)  # Показываем результаты

        case 2:  # Метод Хартли
            rectified_imgL, rectified_imgR = fundamental_matrix_method(imgL, imgR)  # Выполняем метод Хартли

            # Проверяем успешность метода
            if rectified_imgL is None or rectified_imgR is None:
                print("Ошибка метода Хартли.")
                return

            show_results(imgL, imgR, rectified_imgL, rectified_imgR)  # Показываем результаты

        case _:  # Неверный выбор
            print("Ошибка: неверный номер метода.")


# Точка входа в программу
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)  # Создаем папку для сохранения результатов
    main()  # Вызываем главную функцию