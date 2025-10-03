import os
from PIL import Image, ImageEnhance
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

IMAGE_SIZE = 100

def preprocess_image(image_path):

    """Преобразует изображение в нормализованный вектор."""
    img = Image.open(image_path).convert('L')  # Конвертируем в оттенки серого
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = (np.array(img) / 255.0 - 0.5) * 2  # Диапазон от -1 до 1;  Нормализация значений пикселей
    return img_array.flatten()  # Преобразуем в одномерный вектор (100x100 = 10000)


def augment_image(image):
    """Применяет случайные аугментации к изображению и возвращает изменённую версию."""

    # Клонируем изображение, чтобы не менять оригинал
    img = image.copy()

    if np.random.rand() > 0.5:  # 50% шанс на горизонтальный флип
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if np.random.rand() > 0.5:  # 50% шанс на поворот
        img = img.rotate(np.random.randint(-15, 15))  # Случайный угол от -15° до +15°

    if np.random.rand() > 0.5:  # 50% шанс на изменение яркости
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(np.random.uniform(0.7, 1.3))  # Изменяем яркость на 70–130%

    if np.random.rand() > 0.5:  # 50% шанс добавить шум
        img_array = np.array(img) / 255.0  # Преобразуем в массив и нормализуем
        noise = np.random.normal(0, 0.02, img_array.shape)  # Добавляем случайный шум
        img_noisy = np.clip(img_array + noise, 0, 1)  # Обрезаем значения в диапазоне [0,1]
        img = Image.fromarray((img_noisy * 255).astype(np.uint8))  # Конвертируем обратно в изображение

    return img  # Возвращаем изменённое изображение


def resize_and_augment_images(input_dir, output_dir, size=(30, 30), num_augment=2):
    """Увеличивает датасет, применяя аугментацию к изображениям."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        output_folder = os.path.join(output_dir, folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images = os.listdir(folder_path)
        for image_name in images:
            try:
                img_path = os.path.join(folder_path, image_name)
                img = Image.open(img_path)

                # Сохраняем оригинальное изображение
                img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_resized.save(os.path.join(output_folder, image_name))

                # Генерируем дополнительные изображения
                for i in range(num_augment):  # Добавляем несколько копий с изменениями
                    augmented_img = augment_image(img)
                    new_image_name = f"aug_{i}_{image_name}"
                    augmented_img.save(os.path.join(output_folder, new_image_name))

            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")

def load_data(data_dir='new_dataset', test_size=0.2, resize=True, output_dir='resized_dataset'):
    """Загружает данные из указанной директории."""

    # Если нужно, сначала масштабируем изображения
    if resize:
        print("Начинаем масштабирование изображений...")
        resize_and_augment_images(data_dir, output_dir, IMAGE_SIZE)  # увеличиваем 2 раза
        data_dir = output_dir  # Используем resized_dataset как новую директорию

    # Собираем все изображения в единый список с метками
    image_paths = []
    labels = []
    class_names = []

    for idx, folder_name in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        class_names.append(folder_name)
        for image_name in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, image_name))
            labels.append(idx)  # Метка класса

    # Перемешиваем изображения и метки одинаково
    combined = list(zip(image_paths, labels))  # Объединяем пути и метки
    np.random.shuffle(combined)  # Перемешиваем
    image_paths, labels = zip(*combined)  # Разделяем обратно после перемешивания

    # Преобразуем изображения в массивы
    images = []
    for image_path in image_paths:
        try:
            img_vector = preprocess_image(image_path)
            images.append(img_vector)
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")

    # Преобразуем метки в one-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(np.array(labels).reshape(-1, 1))

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(images), y_encoded, test_size=test_size, random_state=42
    )

    return X_train, y_train, X_test, y_test, class_names