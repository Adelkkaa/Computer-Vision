import cv2
import numpy as np

# Параметры для метода усреднения фона
alpha = 0.1  # Скорость обновления фона

# Инициализация переменных для метода усреднения
background_avg = None
background_std = None

# Выбор источника видео (файл или камера)
video_source = "video.mp4"  
cap = cv2.VideoCapture(video_source)

# Параметры для метода кодовой книги (KNN)
fgbg = cv2.createBackgroundSubtractorKNN()

# Параметр для замедления видео (в миллисекундах)
delay = int(1000 / 60) 

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS исходного видео: {fps}")

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
        # Текущая накопленная дисперсия
        background_std = np.zeros_like(gray_frame, dtype="float")

    # Обновляем значения
    cv2.accumulateWeighted(gray_frame, background_avg, alpha)
    cv2.accumulateWeighted((gray_frame - background_avg) ** 2, background_std, alpha)

    # Вычисление стандартного отклонения
    std = cv2.sqrt(background_std)
    # Пороговое значение на основе среднего стандартного отклонения
    diff = cv2.absdiff(gray_frame, background_avg.astype("uint8"))

    # _, fg_mask_avg = cv2.threshold(diff, threshold_value * std, 255, cv2.THRESH_BINARY)
    threshold_value = 2
    adaptive_threshold = threshold_value * std
    adaptive_threshold = np.clip(adaptive_threshold, a_min=10, a_max=255)  # Ограничение порога

    fg_mask_avg = (diff > adaptive_threshold).astype("uint8") * 255

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

    # Замедление воспроизведения видео
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()