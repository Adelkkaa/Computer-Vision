import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Путь к папке с исходными изображениями
input_path = "assets/train"
# Путь к папке для сохранения обработанных изображений
output_path = "assets/train_resized"
os.makedirs(output_path, exist_ok=True)

# Размер, к которому будем приводить изображения
target_size = (512, 512)

# Функция для изменения размера изображения
def resize_image(image_path, output_path, target_size):
    image = cv2.imread(image_path)
    if image is not None:
        resized_image = cv2.resize(image, target_size)
        cv2.imwrite(output_path, resized_image)
    else:
        print(f"Не удалось загрузить изображение: {image_path}")

# Обработка всех изображений в папке
for img_name in os.listdir(input_path):
    img_path = os.path.join(input_path, img_name)
    output_img_path = os.path.join(output_path, img_name)
    resize_image(img_path, output_img_path, target_size)

print("Все изображения изменены до размера 512x512 и сохранены в папку 'assets/train_resized'.")

# Функция для извлечения дескрипторов
def extract_descriptors(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

# Сбор дескрипторов для всех изображений
descriptors_list = []
labels = []

for img_name in os.listdir(output_path):
    img_path = os.path.join(output_path, img_name)
    descriptors = extract_descriptors(img_path)
    if descriptors is not None:
        descriptors_list.append(descriptors)
        # Метка класса: 0 — кошка, 1 — собака
        if img_name.startswith("cat"):
            labels.append(0)
        else:
            labels.append(1)


print(f"Извлечено {len(descriptors_list)} наборов дескрипторов.")

# Объединение всех дескрипторов в одну матрицу
all_descriptors = np.vstack(descriptors_list)


# Количество кластеров (слов в словаре)
n_clusters = 1000  

# Кластеризация с помощью K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(all_descriptors)

print("Словарь построен.")

# Функция для построения гистограммы
def build_histogram(descriptors, kmeans, n_clusters):
    histogram = np.zeros(n_clusters)
    if descriptors is not None:
        cluster_result = kmeans.predict(descriptors)
        for i in cluster_result:
            histogram[i] += 1.0
    return histogram

# Построение гистограмм для всех изображений
X = []
for descriptors in descriptors_list:
    histogram = build_histogram(descriptors, kmeans, n_clusters)
    X.append(histogram)

X = np.array(X)
y = np.array(labels)

print(f"Построено {len(X)} гистограмм.")

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = clf.predict(X_test)

# Классификация тестовых изображений
test_path = "assets/test"
for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)
    descriptors = extract_descriptors(img_path)
    if descriptors is not None:
        histogram = build_histogram(descriptors, kmeans, n_clusters)
        histogram = histogram.reshape(1, -1)  # Преобразуем в 2D-массив для predict

        # Предсказание класса
        predicted_class = clf.predict(histogram)
        if predicted_class[0] == 0:
            print(f"Изображение '{img_name}' классифицировано как 'кошка'.")
        else:
            print(f"Изображение '{img_name}' классифицировано как 'собака'.")
    else:
        print(f"Не удалось извлечь дескрипторы для изображения '{img_name}'.")


# Функция для загрузки изображений
def load_images(class_name, max_count=30):
    images = []
    for img_name in os.listdir(output_path):
        if img_name.startswith(class_name):
            img_path = os.path.join(output_path, img_name)
            img = cv2.imread(img_path)
            images.append(img)
            if len(images) >= max_count:
                break
    return images

# Загрузка изображений кошек и собак
cat_images = load_images("cat")
dog_images = load_images("dog")

# Функция для создания сетки изображений
def create_image_grid(images, grid_size=(5, 6), image_size=(100, 100)):
    rows, cols = grid_size
    grid = np.zeros((rows * image_size[0], cols * image_size[1], 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(images):
                img = cv2.resize(images[idx], image_size)
                grid[i * image_size[0]:(i + 1) * image_size[0],
                     j * image_size[1]:(j + 1) * image_size[1]] = img
    return grid

# Создание сетки для кошек и собак
cat_grid = create_image_grid(cat_images, grid_size=(5, 6), image_size=(100, 100))
dog_grid = create_image_grid(dog_images, grid_size=(5, 6), image_size=(100, 100))

# Отображение сетки изображений
cv2.imshow("Cats Preview", cat_grid)
cv2.imshow("Dogs Preview", dog_grid)



test_images = []
for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)
    img = cv2.imread(img_path)
    test_images.append(img)

# Создание сетки для тестовых изображений
test_grid = create_image_grid(test_images, grid_size=(2, 2), image_size=(200, 200))

# Отображение тестовых изображений
cv2.imshow("Test Images", test_grid)

cv2.waitKey(0)
cv2.destroyAllWindows()