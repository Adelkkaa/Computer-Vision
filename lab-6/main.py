import cv2
import numpy as np
import glob
import os


# Функция калибровки камеры (Метод Богета)
def calibrate_camera(chessboard_images_path, chessboard_size=(7, 11)):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(chessboard_images_path, '*.png'))
    for img_path in images:
        img = cv2.imread(img_path, 0)
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if not objpoints or not imgpoints:
        print("Не удалось найти углы шахматной доски.")
        return None, None

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    return camera_matrix, dist_coeffs


# Функция стереокалибровки (Метод Богета)
def stereo_calibrate(camera_matrix, dist_coeffs, imgL, imgR):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) < 8:
        print("Недостаточно совпадений.")
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    E = camera_matrix.T @ F @ camera_matrix
    _, R, T, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)

    return R, T


# Функция ректификации изображений (Метод Богета)
def rectify_images(imgL, imgR, camera_matrix, dist_coeffs, R, T):
    h, w = imgL.shape
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix, dist_coeffs, camera_matrix, dist_coeffs, (w, h), R, T
    )

    map1L, map2L = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)

    rectified_imgL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
    rectified_imgR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)

    return rectified_imgL, rectified_imgR


# Функция вычисления фундаментальной матрицы и стереоректификации (Метод Хартли)
def fundamental_matrix_method(imgL, imgR):
    # Определяем алгоритм обнаружения ключевых точек
    sift = cv2.SIFT_create()

    # Находим ключевые точки и их описатели
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    # Сопоставляем описатели ключевых точек
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Фильтруем соответствия
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Проверяем, что найдено достаточно совпадений
    if len(good_matches) < 8:
        print("Недостаточно совпадений.")
        return None, None

    # Извлекаем координаты соответствующих точек
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Вычисляем фундаментальную матрицу
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

    # Проверка успешности вычисления фундаментальной матрицы
    if F is None:
        print("Ошибка: не удалось вычислить фундаментальную матрицу.")
        return None, None

    # Вычисляем гомографию
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Применяем гомографию к левому изображению
    rectified_imgL = cv2.warpPerspective(imgL, H, (imgL.shape[1], imgL.shape[0]))

    # Правому изображению не нужно применять преобразование, так как мы выравниваем только левое
    rectified_imgR = imgR

    return rectified_imgL, rectified_imgR


# Функция отображения всех изображений в одном окне
def show_results(imgL, imgR, rectifiedL, rectifiedR):
    imgL_resized = cv2.resize(imgL, (640, 360))
    imgR_resized = cv2.resize(imgR, (640, 360))
    rectifiedL_resized = cv2.resize(rectifiedL, (640, 360))
    rectifiedR_resized = cv2.resize(rectifiedR, (640, 360))

    top_row = np.hstack((imgL_resized, imgR_resized))
    bottom_row = np.hstack((rectifiedL_resized, rectifiedR_resized))
    combined = np.vstack((top_row, bottom_row))

    cv2.imshow("Original (Top) | Rectified (Bottom)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Сохранение изображений
    cv2.imwrite("results/output_image_left.png", rectifiedL)
    cv2.imwrite("results/output_image_right.png", rectifiedR)
    print("Изображения сохранены в папке 'results'.")


# Меню выбора метода
def main():
    print("\nВыберите метод калибровки стереосистемы камер:")
    print("1 - Калибровка методом Богета (с предварительной калибровкой)")
    print("2 - Калибровка методом Хартли (фундаментальная матрица, без калибровки)")
    print("0 - Выход")

    try:
        choice = int(input("Введите номер метода: "))
    except ValueError:
        print("Ошибка: введите число.")
        return

    if choice == 0:
        print("Выход из программы.")
        return

    imgL_path = 'assets/input_image_left.png'
    imgR_path = 'assets/input_image_right.png'
    imgL = cv2.imread(imgL_path, 0)
    imgR = cv2.imread(imgR_path, 0)

    if imgL is None or imgR is None:
        print("Ошибка: не удалось загрузить изображения.")
        return

    match choice:
        case 1:
            chessboard_images_path = 'assets'
            camera_matrix, dist_coeffs = calibrate_camera(chessboard_images_path)

            if camera_matrix is None or dist_coeffs is None:
                print("Ошибка калибровки камеры.")
                return

            R, T = stereo_calibrate(camera_matrix, dist_coeffs, imgL, imgR)
            if R is None or T is None:
                print("Ошибка стереокалибровки.")
                return

            rectified_imgL, rectified_imgR = rectify_images(imgL, imgR, camera_matrix, dist_coeffs, R, T)
            show_results(imgL, imgR, rectified_imgL, rectified_imgR)

        case 2:
            rectified_imgL, rectified_imgR = fundamental_matrix_method(imgL, imgR)
            if rectified_imgL is None or rectified_imgR is None:
                print("Ошибка метода Хартли.")
                return

            show_results(imgL, imgR, rectified_imgL, rectified_imgR)

        case _:
            print("Ошибка: неверный номер метода.")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()