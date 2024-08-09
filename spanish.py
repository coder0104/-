import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

alphabet = list('abcdefghijklmnñopqrstuvwxyz')
char_to_index = {char: i for i, char in enumerate(alphabet)}
index_to_char = {i: char for i, char in enumerate(alphabet)}

def generate_data(num_samples=1000):
    x_data = []
    y_data = []
    for _ in range(num_samples):
        letter1 = np.random.choice(alphabet)
        letter2 = np.random.choice(alphabet)
        x_data.append([char_to_index[letter1], char_to_index[letter2]])
        y_data.append(char_to_index[letter1] * len(alphabet) + char_to_index[letter2])
    return np.array(x_data), np.array(y_data)

x_train, y_train = generate_data()

model = Sequential([
    Embedding(input_dim=len(alphabet), output_dim=10, input_length=2),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(len(alphabet) * len(alphabet), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_loss = []
history_accuracy = []

def test_model():
    test_data, test_labels = generate_data(num_samples=100)
    predictions = model.predict(test_data)
    correct = 0
    for i, prediction in enumerate(predictions):
        predicted_index = np.argmax(prediction)
        if predicted_index == test_labels[i]:
            correct += 1
    accuracy = correct / len(test_data)
    print(f"테스트 정확도: {accuracy * 100:.2f}%")
    return accuracy

def simulate_learning_sessions(target_accuracy=0.95):
    session = 0
    while True:
        session += 1
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        
        history_loss.append(history.history['loss'][0])
        history_accuracy.append(history.history['accuracy'][0])

        test_data, test_labels = generate_data(num_samples=100)
        predictions = model.predict(test_data)
        correct = sum(np.argmax(p) == t for p, t in zip(predictions, test_labels))
        accuracy = correct / len(test_data)
        print(f"세션 {session}: 정확도 = {accuracy * 100:.2f}%")
        
        if accuracy >= target_accuracy or session > 300:
            break
    return session

# Run the simulation
required_sessions = simulate_learning_sessions()
print(f"Required sessions to reach 95% accuracy: {required_sessions}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_loss, label='손실')
plt.xlabel('세션')
plt.ylabel('손실')
plt.title('학습되면서 발생하는 추정 오차로 인해 발생하는 손실')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_accuracy, label='정확도')
plt.xlabel('세션')
plt.ylabel('정확도')
plt.title('세션이 진행되면서의 정확도')
plt.legend()

plt.show()
