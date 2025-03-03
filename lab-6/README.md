### Метод Богета: Калибровка и ректификация стереокамер

Метод Богета — это классический подход к калибровке и ректификации стереокамер, который включает несколько этапов: калибровку камеры, стереокалибровку и ректификацию изображений. Рассмотрим каждый этап подробно.

---

#### 1. **Калибровка камеры**

Калибровка камеры необходима для определения внутренних параметров камеры (матрица камеры и коэффициенты искажений). Эти параметры позволяют корректировать искажения, вызванные оптикой камеры, и переводить координаты пикселей в реальные координаты.

```python
def calibrate_camera(chessboard_images_path, chessboard_size=(7, 11)):
    # Создаем массив точек шахматной доски в 3D-пространстве
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
```

- **`objp`**: Это массив точек шахматной доски в 3D-пространстве. Мы предполагаем, что шахматная доска лежит на плоскости \( Z = 0 \), поэтому координаты \( Z \) равны нулю.
- **`np.mgrid`**: Создает сетку координат для шахматной доски. Например, для доски 7x11 создается 77 точек с координатами \( (0,0), (0,1), \dots, (6,10) \).

```python
    objpoints = []  # Точки в 3D-пространстве
    imgpoints = []  # Точки на изображении

    images = glob.glob(os.path.join(chessboard_images_path, '*.png'))
    for img_path in images:
        img = cv2.imread(img_path, 0)
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
```

- **`cv2.findChessboardCorners`**: Находит углы шахматной доски на изображении. Если углы найдены, они добавляются в `imgpoints`, а соответствующие 3D-точки — в `objpoints`.
- **`objpoints` и `imgpoints`**: Эти массивы используются для калибровки камеры. Они связывают 3D-точки с их проекциями на изображении.

```python
    if not objpoints or not imgpoints:
        print("Не удалось найти углы шахматной доски.")
        return None, None

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    return camera_matrix, dist_coeffs
```

- **`cv2.calibrateCamera`**: Вычисляет матрицу камеры (`camera_matrix`) и коэффициенты искажений (`dist_coeffs`). Эти параметры описывают, как камера проецирует 3D-точки на 2D-изображение.
- **`camera_matrix`**: Матрица 3x3, содержащая фокусные расстояния \( f_x, f_y \) и координаты оптического центра \( c_x, c_y \).
- **`dist_coeffs`**: Коэффициенты искажений, такие как радиальное и тангенциальное искажение.

---

#### 2. **Стереокалибровка**

Стереокалибровка определяет относительное положение и ориентацию двух камер (матрицы поворота \( R \) и вектора смещения \( T \)).

```python
def stereo_calibrate(camera_matrix, dist_coeffs, imgL, imgR):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)
```

- **`cv2.SIFT_create()`**: Используется для обнаружения ключевых точек и их дескрипторов на изображениях. SIFT (Scale-Invariant Feature Transform) устойчив к изменениям масштаба и поворота.
- **`kp1, des1`**: Ключевые точки и их дескрипторы для левого изображения.
- **`kp2, des2`**: Ключевые точки и их дескрипторы для правого изображения.

```python
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
```

- **`cv2.BFMatcher()`**: Используется для сопоставления дескрипторов ключевых точек между изображениями.
- **`good_matches`**: Фильтрация совпадений по расстоянию. Мы оставляем только те совпадения, где расстояние между дескрипторами меньше 70% от следующего ближайшего совпадения.

```python
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    E = camera_matrix.T @ F @ camera_matrix
    _, R, T, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)
```

- **`cv2.findFundamentalMat`**: Вычисляет фундаментальную матрицу \( F \), которая связывает соответствующие точки на двух изображениях.
- **`E = camera_matrix.T @ F @ camera_matrix`**: Вычисляет существенную матрицу \( E \) из фундаментальной матрицы и матрицы камеры.
- **`cv2.recoverPose`**: Восстанавливает матрицу поворота \( R \) и вектор смещения \( T \) из существенной матрицы.

---

#### 3. **Ректификация изображений**

Ректификация выравнивает изображения так, чтобы соответствующие точки находились на одной горизонтальной линии. Это упрощает поиск соответствий и вычисление глубины.

```python
def rectify_images(imgL, imgR, camera_matrix, dist_coeffs, R, T):
    h, w = imgL.shape
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix, dist_coeffs, camera_matrix, dist_coeffs, (w, h), R, T
    )
```

- **`cv2.stereoRectify`**: Вычисляет матрицы поворота \( R1, R2 \) и проекции \( P1, P2 \) для ректификации изображений.

```python
    map1L, map2L = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)

    rectified_imgL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
    rectified_imgR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)
```

- **`cv2.initUndistortRectifyMap`**: Создает карты преобразования для ректификации.
- **`cv2.remap`**: Применяет карты преобразования к изображениям, чтобы выровнять их.

---

### Метод Хартли: Ректификация без калибровки

Метод Хартли основан на вычислении фундаментальной матрицы и гомографии для ректификации изображений без предварительной калибровки камеры. Этот метод менее точен, чем метод Богета, но проще в реализации.

---

#### 1. **Вычисление фундаментальной матрицы**

```python
def fundamental_matrix_method(imgL, imgR):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)
```

- **`cv2.SIFT_create()`**: Обнаружение ключевых точек и их дескрипторов.

```python
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
```

- **Фильтрация совпадений**: Оставляем только качественные совпадения.

```python
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
```

- **`cv2.findFundamentalMat`**: Вычисляет фундаментальную матрицу \( F \), которая связывает соответствующие точки на двух изображениях.

---

#### 2. **Вычисление гомографии**

```python
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
```

- **`cv2.findHomography`**: Вычисляет гомографию \( H \), которая преобразует одно изображение в плоскость другого.

---

#### 3. **Ректификация изображений**

```python
    rectified_imgL = cv2.warpPerspective(imgL, H, (imgL.shape[1], imgL.shape[0]))
    rectified_imgR = imgR
```

- **`cv2.warpPerspective`**: Применяет гомографию к левому изображению, чтобы выровнять его с правым.

---

### Заключение

- **Метод Богета** требует предварительной калибровки камеры и дает более точные результаты. Он подходит для задач, где важна точность, например, в 3D-реконструкции.
- **Метод Хартли** проще в реализации, но менее точен. Он подходит для задач, где калибровка камеры невозможна или не требуется.
