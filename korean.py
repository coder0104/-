import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM  # LSTMV1이 아닌 LSTM 사용

import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

jamos = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
jamo_to_index = {jamo: i for i, jamo in enumerate(jamos)}
index_to_jamo = {i: jamo for i, jamo in enumerate(jamos)}

def generate_data(num_samples=1000):
    x_data = []
    y_data = []
    for _ in range(num_samples):
        consonant = np.random.choice(jamos[:14])
        vowel = np.random.choice(jamos[14:])
        x_data.append([jamo_to_index[consonant], jamo_to_index[vowel]])
        y_data.append(jamo_to_index[consonant] * 14 + jamo_to_index[vowel])
    return np.array(x_data), np.array(y_data)

x_train, y_train = generate_data()

model = Sequential([
    Embedding(input_dim=len(jamos), output_dim=10, input_length=2),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(len(jamos) * len(jamos), activation='softmax')
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
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
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
        print(f"Session {session}: Accuracy = {accuracy * 100:.2f}%")
        
        if accuracy >= target_accuracy:
            break
    return session

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
