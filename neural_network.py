import pickle
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.weights = []
        self.biases = []
        self.initialize_weights()
        self.losses = []
        self.reverse_label_mapping = None  # Добавляем поле для обратного сопоставления

    def initialize_weights(self):
        """Инициализирует веса и смещения случайными значениями."""
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, sizes[i + 1])))

    def plot_losses(self):
        """Строит график потерь."""
        plt.plot(self.losses)
        plt.xlabel("Эпоха")
        plt.ylabel("Потери")
        plt.title("График потерь во время обучения")
        plt.show()

    def save_model(self, file_path, class_names=None):
        """Сохраняет модель в файл вместе с названиями классов."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        if class_names:
            with open("class_names.pkl", "wb") as f:
                pickle.dump(class_names, f)  # Сохраняем список классов
        print(f"Модель и названия классов сохранены.")

    @staticmethod
    def load_model(file_path):
        """Загружает модель из файла."""
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        try:
            with open("class_names.pkl", "rb") as f:
                class_names = pickle.load(f)  # Загружаем список классов
            model.class_names = class_names  # Добавляем список классов в модель
        except FileNotFoundError:
            model.class_names = None  # Если файл не найден, устанавливаем None
        return model

    def activate(self, x):
        """Применяет функцию активации."""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'arctan':
            return np.arctan(x)

    def forward(self, X):
        """Выполняет прямое распространение."""
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            dropout_rate = 0.2  # 20% нейронов случайно отключаются во время обучения
            a = self.activate(z)

            if self.training:  # Только во время тренировки
                dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=a.shape) / (1 - dropout_rate)
                a *= dropout_mask  # Обнуляем часть нейронов

            activations.append(a)

        return activations

    def backward(self, X, y, learning_rate):
        """Выполняет обратное распространение ошибки."""
        activations = self.forward(X)

        # Убедитесь, что размерности совпадают
        if activations[-1].shape != y.shape:
            raise ValueError(
                f"Размерности не совпадают: activations[-1].shape={activations[-1].shape}, y.shape={y.shape}")

        error = activations[-1] - y
        deltas = [error * activations[-1] * (1 - activations[-1])]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * activations[i] * (1 - activations[i])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            l2_lambda = 0.001  # Коэффициент регуляризации (можно попробовать 0.0005 - 0.01)
            self.weights[i] -= learning_rate * (np.dot(activations[i].T, deltas[i]) + l2_lambda * self.weights[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X, y, epochs, learning_rate):
        self.training = True
        """Обучает нейронную сеть."""
        for epoch in range(epochs):
            epoch_loss = 0  # Суммарная потеря для текущей эпохи
            for i in range(len(X)):
                self.backward(X[i].reshape(1, -1), y[i].reshape(1, -1), learning_rate)
                epoch_loss += self.calculate_loss(X[i].reshape(1, -1), y[i].reshape(1, -1))
            avg_loss = epoch_loss / len(X)  # Средняя потеря для эпохи
            self.losses.append(avg_loss)  # Сохраняем потерю

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def calculate_loss(self, X, y):
        """Вычисляет среднюю квадратичную ошибку."""
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)

    def predict(self, X):
        self.training = False
        """Выполняет предсказание для входных данных."""
        return self.forward(X)[-1]
