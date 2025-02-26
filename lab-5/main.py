import cv2
import numpy as np

# Загрузка видео или подключение к камере
video_input = cv2.VideoCapture('video.mp4')  # Для видеофайла

# Список для хранения точек, которые нужно отслеживать
selected_points = []

# Функция-обработчик кликов мыши
def handle_mouse_click(event, x_coord, y_coord, flags, params):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x_coord, y_coord))  # Добавляем координаты клика в список точек
        print(f'Добавлена точка: {(x_coord, y_coord)}')

# Назначение функции обработчика на окно
cv2.namedWindow('Optical Flow Tracking')
cv2.setMouseCallback('Optical Flow Tracking', handle_mouse_click)

# Маска для отрисовки траекторий
trajectory_overlay = None

# Параметры для алгоритма Лукаса-Канаде
optical_flow_params = dict(
    winSize=(15, 15),  # Размер окна для поиска
    maxLevel=2,  # Количество уровней пирамиды
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Критерии остановки
)

# Основной цикл обработки видео
while True:
    # Чтение кадра из видео
    ret, current_frame = video_input.read()
    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Инициализация маски при первом кадре
    if trajectory_overlay is None:
        trajectory_overlay = np.zeros_like(current_frame)

    # Если есть точки для отслеживания
    if len(selected_points) > 0:
        # Преобразование точек в формат для OpenCV
        initial_points = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)

        # Если это не первый кадр, вычисляем оптический поток
        if 'previous_gray_frame' in locals():
            # Вычисляем оптический поток
            new_points, status, errors = cv2.calcOpticalFlowPyrLK(
                previous_gray_frame, gray_frame, initial_points, None, **optical_flow_params
            )

            # Отбираем только успешно найденные точки
            valid_new_points = new_points[status == 1]
            valid_initial_points = initial_points[status == 1]

            # Отрисовка траекторий движения точек
            for i, (new, old) in enumerate(zip(valid_new_points, valid_initial_points)):
                new_x, new_y = new.ravel().astype(int)
                old_x, old_y = old.ravel().astype(int)
                # Рисуем линию на маске
                trajectory_overlay = cv2.line(trajectory_overlay, (new_x, new_y), (old_x, old_y), (0, 255, 0), 2)
                # Рисуем текущую позицию точки
                current_frame = cv2.circle(current_frame, (new_x, new_y), 5, (0, 0, 255), -1)

            # Обновляем точки для следующего кадра
            selected_points = valid_new_points.reshape(-1, 2).tolist()

    # Наложение маски с траекториями на текущий кадр
    result_frame = cv2.add(current_frame, trajectory_overlay)

    # Отображение результата
    cv2.imshow('Optical Flow Tracking', result_frame)

    # Обновляем предыдущий кадр
    previous_gray_frame = gray_frame.copy()

    # Выход по нажатию ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Освобождение ресурсов
video_input.release()
cv2.destroyAllWindows()