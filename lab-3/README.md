### Теоретическая часть лабораторной работы по сегментации изображений

#### 1. Введение
Сегментация изображений — это процесс разделения изображения на несколько частей (сегментов), которые соответствуют различным объектам или областям интереса. Основная цель сегментации — упрощение или изменение представления изображения для более легкого анализа. Сегментация используется в различных областях, таких как медицинская визуализация, распознавание объектов, анализ спутниковых снимков и т.д.

#### 2. Методы сегментации

##### 2.1. Сегментация по контурам
Метод основан на обнаружении границ между объектами. Для этого используются различные операторы, такие как Собель, Превитт, Лаплас и др. После обнаружения границ контуры связываются и формируются области.

**Реализация в коде:**
1. Преобразование изображения в оттенки серого:
   ```python
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   ```
2. Применение размытия по Гауссу для уменьшения шума:
   ```python
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   ```
3. Использование оператора Собеля для выделения границ:
   ```python
   sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
   sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
   sobel_combined = cv2.magnitude(sobel_x, sobel_y)
   sobel_combined = np.uint8(sobel_combined)
   ```
4. Пороговое преобразование для выделения контуров:
   ```python
   ret, edges = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
   ```
5. Нахождение и отрисовка контуров:
   ```python
   contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   output = image.copy()
   cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
   ```

##### 2.2. Морфологическая сегментация (метод водоразделов)
Метод основан на интерпретации изображения как топографической поверхности. Локальные минимумы рассматриваются как центры областей, а границы между ними — как водоразделы. В OpenCV для этого используется функция `cv2.watershed()`.

**Реализация в коде:**
1. Инициализация маркеров:
   ```python
   markers = np.zeros(image.shape[:2], dtype=np.int32)
   ```
2. Обработка событий мыши для разметки объектов и фона:
   ```python
   def mouse_callback(event, x, y, flags, param):
       global drawing, markers, marker_image, current_marker
       if event == cv2.EVENT_LBUTTONDOWN:  # Левый клик — объект
           drawing = True
           current_marker = 1
           cv2.circle(marker_image, (x, y), 5, (255, 255, 255), -1)
           cv2.circle(markers, (x, y), 5, current_marker, -1)
       elif event == cv2.EVENT_RBUTTONDOWN:  # Правый клик — фон
           drawing = True
           current_marker = 2
           cv2.circle(marker_image, (x, y), 5, (0, 0, 255), -1)
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
   ```
3. Применение метода водоразделов:
   ```python
   markers = cv2.watershed(image, markers)
   ```
4. Отображение результата сегментации:
   ```python
   washed = np.zeros_like(image, dtype=np.uint8)
   washed[markers == 1] = [0, 255, 0]  # Объекты (зеленый)
   washed[markers == 2] = [255, 0, 0]  # Фон (синий)
   washed[markers == -1] = [0, 0, 255]  # Границы (красный)
   ```

##### 2.3. Пороговая сегментация
Пороговая сегментация делит изображение на две части: объекты и фон, на основе порогового значения.

**2.3.1. Глобальный порог**
Все пиксели изображения сравниваются с одним пороговым значением. Если значение пикселя выше порога, он относится к объекту, иначе — к фону.

**Реализация в коде:**
1. Преобразование изображения в оттенки серого:
   ```python
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   ```
2. Применение глобального порога:
   ```python
   ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
   ```

**2.3.2. Адаптивный порог**
Пороговое значение вычисляется для каждого пикселя на основе локальной области вокруг него. Это полезно для изображений с неравномерной освещенностью.

**Реализация в коде:**
1. Преобразование изображения в оттенки серого:
   ```python
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   ```
2. Применение адаптивного порога:
   ```python
   adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
   ```

##### 2.4. Сегментация методом K-средних
Метод основан на кластеризации пикселей в пространстве признаков (например, цветовых каналов). Пиксели группируются в кластеры, которые соответствуют различным объектам или областям.

**Реализация в коде:**
1. Преобразование изображения в массив пикселей:
   ```python
   Z = image.reshape((-1, 3))
   Z = np.float32(Z)
   ```
2. Применение алгоритма K-средних:
   ```python
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
   ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
   ```
3. Преобразование результата в изображение:
   ```python
   center = np.uint8(center)
   res = center[label.flatten()]
   res2 = res.reshape((image.shape))
   ```
4. Определение цветов для объекта и фона:
   ```python
   if center[0].mean() < center[1].mean():
       object_color = np.array([255, 0, 0], dtype=np.uint8)
       background_color = np.array([0, 0, 255], dtype=np.uint8)
   else:
       object_color = np.array([0, 0, 255], dtype=np.uint8)
       background_color = np.array([255, 0, 0], dtype=np.uint8)
   segmented_image = np.zeros_like(image, dtype=np.uint8)
   segmented_image[label.reshape(image.shape[:2]) == 0] = object_color
   segmented_image[label.reshape(image.shape[:2]) == 1] = background_color
   ```

##### 2.5. Сегментация методом поиска минимальных разрезов графа (GrabCut)
Метод основан на моделировании изображения как графа, где пиксели — это вершины, а ребра — связи между ними. Алгоритм находит минимальный разрез графа, который разделяет объект и фон.

**Реализация в коде:**
1. Инициализация маски и моделей:
   ```python
   mask = np.zeros(image.shape[:2], np.uint8)
   bgd_model = np.zeros((1, 65), np.float64)
   fgd_model = np.zeros((1, 65), np.float64)
   ```
2. Применение алгоритма GrabCut:
   ```python
   rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)
   cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
   ```
3. Создание маски для отображения результата:
   ```python
   mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
   ```
4. Применение маски к изображению:
   ```python
   output = image * mask2[:, :, np.newaxis]
   ```

#### 3. Основная функция программы
Основная функция программы позволяет пользователю выбрать метод сегментации и применить его к изображению. В зависимости от выбранного метода, программа запрашивает необходимые параметры и отображает результат.

**Реализация в коде:**
1. Загрузка изображения:
   ```python
   image = cv2.imread(filename)
   ```
2. Выбор метода сегментации:
   ```python
   if method == 1:
       result = contour_segmentation(image)
       show_images("Original Image", image, "Contour Segmentation", result)
   elif method == 2:
       markers = np.zeros(image.shape[:2], dtype=np.int32)
       marker_image = image.copy()
       cv2.namedWindow("Image")
       cv2.setMouseCallback("Image", mouse_callback)
       print("Левый клик — объект, правый клик — фон. Нажмите 'q' для сегментации.")
       while True:
           cv2.imshow("Image", marker_image)
           key = cv2.waitKey(1) & 0xFF
           if key == ord('q'):
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
   ```

#### 4. Интерактивный функционал для водоразделов
Для метода водоразделов реализован интерактивный функционал, позволяющий пользователю отмечать объекты и фон с помощью кликов мыши.

**Реализация в коде:**
1. Обработка событий мыши:
   ```python
   def mouse_callback(event, x, y, flags, param):
       global drawing, markers, marker_image, current_marker
       if event == cv2.EVENT_LBUTTONDOWN:  # Левый клик — объект
           drawing = True
           current_marker = 1
           cv2.circle(marker_image, (x, y), 5, (255, 255, 255), -1)
           cv2.circle(markers, (x, y), 5, current_marker, -1)
       elif event == cv2.EVENT_RBUTTONDOWN:  # Правый клик — фон
           drawing = True
           current_marker = 2
           cv2.circle(marker_image, (x, y), 5, (0, 0, 255), -1)
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
   ```
2. Отображение маркеров на изображении:
   ```python
   cv2.circle(marker_image, (x, y), 5, (255, 255, 255), -1)
   cv2.circle(markers, (x, y), 5, current_marker, -1)
   ```
3. Запуск сегментации по нажатию клавиши `q`:
   ```python
   while True:
       cv2.imshow("Image", marker_image)
       key = cv2.waitKey(1) & 0xFF
       if key == ord('q'):
           break
   ```

#### 5. Заключение
Программа предоставляет возможность выбора различных методов сегментации изображений, включая интерактивный метод водоразделов. Каждый метод подробно описан и реализован с использованием библиотеки OpenCV. Программа позволяет пользователю экспериментировать с различными подходами к сегментации и визуализировать результаты.