import cv2
import numpy as np

# Параметры для метода усреднения фона
alpha = 0.1  # Скорость обновления фона

# Инициализация переменных для метода усреднения
background_avg = None
background_std = None

# Выбор источника видео (файл или камера)
video_source = "video.mp4"  # Замените на 0 для использования камеры
cap = cv2.VideoCapture(video_source)

# Параметры для метода кодовой книги (KNN)
fgbg = cv2.createBackgroundSubtractorKNN()

# Параметр для замедления видео (в миллисекундах)
delay = 40  # Увеличьте это значение для более медленного воспроизведения

# Создаем объект для записи видео (если нужно сохранить результат)
output_video = cv2.VideoWriter(
    'output_video.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    30, 
    (640, 360)
)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Преобразование в градации серого для упрощения вычислений
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- Метод 1: Усреднение фона ----
    if background_avg is None:
        # Среднее значение фона
        background_avg = gray_frame.astype("float")
        # Текущее стандартное отклонение
        background_std = np.zeros_like(gray_frame, dtype="float")

    # Обновляем среднее и стандартное отклонение
    cv2.accumulateWeighted(gray_frame, background_avg, alpha)
    cv2.accumulateWeighted((gray_frame - background_avg) ** 2, background_std, alpha)

    # Вычисление стандартного отклонения
    std = cv2.sqrt(background_std)

    # Пороговое значение на основе среднего стандартного отклонения
    mean_std = np.mean(std)
    diff = cv2.absdiff(gray_frame, background_avg.astype("uint8"))
    _, fg_mask_avg = cv2.threshold(diff, 2 * mean_std, 255, cv2.THRESH_BINARY)

    # ---- Метод 2: Кодовая книга (KNN) ----
    fg_mask_cb = fgbg.apply(frame)

    # Уменьшение размеров окон
    fg_mask_avg_resized = cv2.resize(fg_mask_avg, (640, 360))  # Уменьшаем до 640x360
    fg_mask_cb_resized = cv2.resize(fg_mask_cb, (640, 360))  # Уменьшаем до 640x360
    frame_resized = cv2.resize(frame, (640, 360))  # Уменьшаем исходное видео до 640x360

    # Добавляем текстовые пояснения
    cv2.putText(frame_resized, "Original Video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(fg_mask_avg_resized, "Foreground (Average + Std Dev)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(fg_mask_cb_resized, "Foreground (Codebook - KNN)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Отображение результата
    cv2.imshow("Original Video", frame_resized)  # Отображаем исходное видео
    cv2.imshow("Foreground (Average + Std Dev)", fg_mask_avg_resized)
    cv2.imshow("Foreground (Codebook - KNN)", fg_mask_cb_resized)

    # Сохранение результата в видеофайл (если нужно)
    combined_frame = np.hstack((frame_resized, cv2.cvtColor(fg_mask_avg_resized, cv2.COLOR_GRAY2BGR), cv2.cvtColor(fg_mask_cb_resized, cv2.COLOR_GRAY2BGR)))
    output_video.write(combined_frame)

    # Замедление воспроизведения видео
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
output_video.release()
cv2.destroyAllWindows()