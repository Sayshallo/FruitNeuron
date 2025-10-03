from neural_network import NeuralNetwork
from data_preprocessing import load_data
import numpy as np

def main():
    # Параметры сети
    IMAGE_SIZE = 100  # Размер стороны квадратного изображения
    input_size = IMAGE_SIZE * IMAGE_SIZE  # Автоматический расчёт размера входного вектора
    hidden_sizes = [512, 256]  # Количества нейронов в скрытых слоях (их 2)
    output_size = 5  # Количество классов (например: яблоко, банан, апельсин, груша, виноград)
    activation = 'sigmoid'  # Функция активации

    # Загрузка данных
    X_train, y_train, X_test, y_test, class_names = load_data()

    # Создание нейронной сети
    nn = NeuralNetwork(input_size, hidden_sizes, output_size, activation)

    # Проверка размерностей данных
    print(f"Размерность X_train: {X_train.shape}")
    print(f"Размерность y_train: {y_train.shape}")
    print(f"Размерность weights[0]: {nn.weights[0].shape}")
    print(f"Размерность biases[0]: {nn.biases[0].shape}")

    # Обучение сети
    epochs = 200
    learning_rate = 0.05
    nn.train(X_train, y_train, epochs, learning_rate)

    # Тестирование сети
    test_accuracy = evaluate_accuracy(nn, X_test, y_test)
    print(f"Точность на тестовых данных: {test_accuracy * 100:.2f}%")

    # Построение графика потерь
    nn.plot_losses()

    nn.save_model("trained_model.pkl", class_names=class_names)

def evaluate_accuracy(nn, X, y):
    correct = 0
    for i in range(len(X)):
        prediction = np.argmax(nn.predict(X[i]))
        true_label = np.argmax(y[i])
        if prediction == true_label:
            correct += 1
    return correct / len(X)

if __name__ == "__main__":
    main()
