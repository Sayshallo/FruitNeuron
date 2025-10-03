import tkinter as tk
from tkinter import filedialog, Listbox
from PIL import Image, ImageTk
import numpy as np
import os  # Импортируем os для работы с файловой системой
from neural_network import NeuralNetwork  # Импортируем нашу нейронную сеть
from data_preprocessing import preprocess_image

IMAGE_SIZE = 100

class ImageRecognitionApp:
    def __init__(self, root, model, dataset_dir="new_dataset"):
        self.root = root
        self.model = model  # Принимаем модель как параметр
        self.dataset_dir = dataset_dir  # Директория датасета
        self.root.title("Распознавание изображений")
        self.root.geometry("600x600")  # Увеличим размер окна

        # Кнопка для выбора файла
        self.select_button = tk.Button(root, text="Выбрать изображение", command=self.load_image)
        self.select_button.pack(pady=20)

        # Метка для отображения изображения
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Метка для вывода результата
        self.result_label = tk.Label(root, text="Результат будет здесь", font=("Arial", 14))
        self.result_label.pack(pady=20)

        # Отображаем список доступных классов
        self.class_list_label = tk.Label(root, text="Доступные классы:", font=("Arial", 12))
        self.class_list_label.pack(pady=10)

        # Создаем ListBox для отображения списка классов
        self.class_listbox = Listbox(root, height=10, width=30, font=("Arial", 12))
        self.class_listbox.pack(pady=10)

        # Заполняем ListBox названиями папок из датасета
        self.load_class_names()

    def load_class_names(self):
        """Загружает названия папок из директории датасета."""
        if os.path.exists(self.dataset_dir):  # Проверяем существование директории
            class_names = [
                folder_name for folder_name in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, folder_name))
            ]
            for class_name in class_names:
                self.class_listbox.insert(tk.END, class_name)  # Добавляем название папки в ListBox
        else:
            self.class_listbox.insert(tk.END, "Датасет не найден")

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Отображаем изображение
            img = Image.open(file_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            # Обрабатываем изображение
            input_data = preprocess_image(file_path).reshape(1, -1)

            # Проверяем, что модель загружена
            if self.model is None:
                self.result_label.config(text="Модель не загружена")
                return

            # Выполняем предсказание
            prediction = self.model.predict(input_data)
            predicted_class_index = np.argmax(prediction)

            # Выводим результат
            if hasattr(self.model, "class_names") and self.model.class_names:
                predicted_class_name = self.model.class_names[predicted_class_index]
            else:
                predicted_class_name = f"Класс {predicted_class_index}"

            self.result_label.config(text=f"Предсказание: {predicted_class_name}")

if __name__ == "__main__":
    # Загружаем предварительно обученную модель
    try:
        model = NeuralNetwork.load_model("trained_model.pkl")
    except FileNotFoundError:
        print("Файл модели не найден. Сначала обучите модель.")
        exit()

    # Создаем GUI
    root = tk.Tk()
    app = ImageRecognitionApp(root, model=model)  # Передаём модель в конструктор
    root.mainloop()