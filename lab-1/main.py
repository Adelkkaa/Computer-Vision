import cv2
import numpy as np


# Загрузка изображений-шаблонов с проверкой
template_sh = cv2.imread('./assets/sh.png', cv2.IMREAD_GRAYSCALE)
template_a = cv2.imread('./assets/a.png', cv2.IMREAD_GRAYSCALE)
if template_sh is None or template_a is None:
    print("Ошибка загрузки изображений шаблонов.")
    exit()

# Бинаризация изображений шаблонов
_, thresh_sh = cv2.threshold(template_sh, 30, 255, cv2.THRESH_BINARY_INV)
_, thresh_a = cv2.threshold(template_a, 30, 255, cv2.THRESH_BINARY_INV)

# Загрузка входного изображения с проверкой
img = cv2.imread('./assets/main.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Ошибка загрузки входного изображения.")
    exit()

# Бинаризация входного изображения
_, thresh_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)

# Поиск контуров на изображениях шаблонов
contours_sh, _ = cv2.findContours(thresh_sh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_a, _ = cv2.findContours(thresh_a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Проверим, есть ли контуры в шаблонах
if len(contours_a) == 0:
    print("Контур для буквы 'А' не найден.")
    exit()

if len(contours_sh) == 0:
    print("Контур для буквы 'Ш' не найден.")
    exit()

# Поиск контуров на входном изображении
contours_img, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Функция для сравнения контуров с использованием инвариантных моментов
def compare_contours(template_contour, input_contours, threshold=0.9, min_area=150, max_area=10000):
    similar_contours = []
    for contour in input_contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:  # Фильтрация по площади
            match_value = cv2.matchShapes(template_contour, contour, cv2.CONTOURS_MATCH_I1, 0)
            if match_value < threshold:  # Порог совпадения
                similar_contours.append((contour, match_value))
    return similar_contours

# Поиск похожих контуров для буквы "Ш"
similar_contours_sh = compare_contours(contours_sh[0], contours_img)

# Поиск похожих контуров для буквы "Д"
similar_contours_a = compare_contours(contours_a[0], contours_img)

# Создание выходного изображения с похожими контурами
output_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # Черное изображение для отрисовки контуров

# Отрисовка контуров буквы "Ш" синим цветом
for contour, match_value in similar_contours_sh:
    cv2.drawContours(output_img, [contour], -1, (255, 0, 0), 1)  # Синий цвет (BGR), толщина 1 пиксель
    print(f'Контур похож на "Ш" с параметром совпадения: {match_value}')

# Отрисовка контуров буквы "Д" зелёным цветом
for contour, match_value in similar_contours_a:
    cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 1)  # Зелёный цвет (BGR), толщина 1 пиксель
    print(f'Контур похож на "А" с параметром совпадения: {match_value}')


# Вывод изображений с контуром букв
cv2.namedWindow('Template D', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Image', 900, 700)
cv2.imshow('original', img)

cv2.namedWindow('Template Matched Contours', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Matched Contours', 900, 700)
cv2.imshow('Matched Contours', output_img)

# Вывод шаблонных букв в отдельных окнах с черным фоном и контуром
template_sh_img = np.zeros_like(thresh_sh)
cv2.drawContours(template_sh_img, contours_sh, -1, 255, 2)  # Белые контуры на черном фоне

template_a_img = np.zeros_like(thresh_a)
cv2.drawContours(template_a_img, contours_a, -1, 255, 2)  # Белые контуры на черном фоне

cv2.namedWindow('Template SH', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Template SH', 300, 300)
cv2.imshow('Template SH', template_sh_img)

cv2.namedWindow('Template A', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Template A', 300, 300)
cv2.imshow('Template A',template_a_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
