## **Теория и реализация лабораторной работы**

### **1. Цель работы**
Цель лабораторной работы — разработать систему классификации изображений на основе подхода **Bag-of-Words (BoW)**. Этот подход позволяет описывать изображения с помощью гистограмм, построенных на основе ключевых точек и их дескрипторов.

---

### **2. Основные шаги алгоритма Bag-of-Words**
1. **Детектирование ключевых точек и вычисление дескрипторов**.
2. **Построение словаря (Bag-of-Words)** с помощью кластеризации дескрипторов.
3. **Построение гистограмм** для каждого изображения.
4. **Обучение классификатора** на гистограммах.
5. **Классификация новых изображений**.

---

### **3. Теория каждого шага**

#### **3.1. Детектирование ключевых точек и вычисление дескрипторов**
- **Ключевые точки** — это особые точки на изображении, которые характеризуют уникальные особенности (например, углы, границы).
- **Дескрипторы** — это числовые векторы, которые описывают локальную область вокруг ключевой точки.
- В работе используется алгоритм **ORB** (Oriented FAST and Rotated BRIEF), который сочетает детектор ключевых точек (FAST) и дескриптор (BRIEF).

#### **3.2. Построение словаря (Bag-of-Words)**
- Все дескрипторы из всех изображений объединяются в одну матрицу.
- На этой матрице выполняется кластеризация (например, с помощью **K-means**).
- Каждый кластер представляет собой "слово" в словаре. Центр кластера (центроид) — это "слово".
- Количество кластеров (\( K \)) задается заранее (например, 100).

#### **3.3. Построение гистограмм**
- Для каждого изображения строится гистограмма, которая показывает, сколько дескрипторов изображения попало в каждый кластер.
- Гистограмма нормализуется, чтобы сделать её инвариантной к размеру изображения.

#### **3.4. Обучение классификатора**
- Гистограммы всех изображений из обучающей выборки используются как признаки для обучения классификатора.
- В работе используется **SVM** (Support Vector Machine), который учится разделять классы на основе гистограмм.

#### **3.5. Классификация новых изображений**
- Для нового изображения выполняются те же шаги: детектирование ключевых точек, вычисление дескрипторов, построение гистограммы.
- Гистограмма подается в обученный классификатор, который предсказывает класс изображения.

---

### **4. Реализация**

#### **4.1. Предобработка изображений**
Все изображения приводятся к одному размеру (например, 512x512), чтобы упростить дальнейшую обработку.

```python
import cv2
import os

input_path = "assets/train"
output_path = "assets/train_resized"
os.makedirs(output_path, exist_ok=True)
target_size = (512, 512)

def resize_image(image_path, output_path, target_size):
    image = cv2.imread(image_path)
    if image is not None:
        resized_image = cv2.resize(image, target_size)
        cv2.imwrite(output_path, resized_image)
    else:
        print(f"Не удалось загрузить изображение: {image_path}")

for img_name in os.listdir(input_path):
    img_path = os.path.join(input_path, img_name)
    output_img_path = os.path.join(output_path, img_name)
    resize_image(img_path, output_img_path, target_size)

print("Все изображения изменены до размера 512x512.")
```

---

#### **4.2. Извлечение ключевых точек и дескрипторов**
Для каждого изображения извлекаются ключевые точки и дескрипторы с помощью ORB.

```python
def extract_descriptors(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

descriptors_list = []
labels = []

for img_name in os.listdir(output_path):
    img_path = os.path.join(output_path, img_name)
    descriptors = extract_descriptors(img_path)
    if descriptors is not None:
        descriptors_list.append(descriptors)
        labels.append(0 if img_name.startswith("cat") else 1)

print(f"Извлечено {len(descriptors_list)} наборов дескрипторов.")
```

---

#### **4.3. Построение словаря**
Все дескрипторы объединяются и кластеризуются с помощью K-means.

```python
from sklearn.cluster import KMeans

all_descriptors = np.vstack(descriptors_list)
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(all_descriptors)

print("Словарь построен.")
```

---

#### **4.4. Построение гистограмм**
Для каждого изображения строится гистограмма.

```python
def build_histogram(descriptors, kmeans, n_clusters):
    histogram = np.zeros(n_clusters)
    if descriptors is not None:
        cluster_result = kmeans.predict(descriptors)
        for i in cluster_result:
            histogram[i] += 1.0
    return histogram

X = []
for descriptors in descriptors_list:
    histogram = build_histogram(descriptors, kmeans, n_clusters)
    X.append(histogram)

X = np.array(X)
y = np.array(labels)

print(f"Построено {len(X)} гистограмм.")
```

---

#### **4.5. Обучение классификатора**
На гистограммах обучается SVM.

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность классификатора: {accuracy * 100:.2f}%")
```

---

#### **4.6. Классификация новых изображений**
Новые изображения классифицируются с помощью обученной модели.

```python
test_path = "assets/test"
for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)
    descriptors = extract_descriptors(img_path)
    if descriptors is not None:
        histogram = build_histogram(descriptors, kmeans, n_clusters)
        histogram = histogram.reshape(1, -1)
        predicted_class = clf.predict(histogram)
        print(f"Изображение '{img_name}' классифицировано как {'кошка' if predicted_class[0] == 0 else 'собака'}.")
    else:
        print(f"Не удалось извлечь дескрипторы для изображения '{img_name}'.")
```
