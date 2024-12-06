import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '8'
tf.config.threading.set_inter_op_parallelism_threads(8)

def load_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28, 28)


def load_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convertim la grayscale
    img_resized = img.resize((28, 28))  # Redimensionăm la 28x28
    img_inverted = 255 - np.array(img_resized)  # Inversăm valorile
    img_normalized = img_inverted / 255.0  # Normalizăm valorile între 0 și 1

    # Afișăm imaginea preprocesată
    plt.imshow(img_normalized, cmap="gray")
    plt.title("Imagine preprocesată")
    plt.show()

    return img_normalized

def predict_image():
    while True:
        print("\nSelectați o imagine de test (sau apăsați Anulare pentru a ieși).")
        root = tk.Tk()
        root.withdraw()  # Ascundem fereastra principală Tkinter
        file_path = filedialog.askopenfilename(title="Selectați o imagine",
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            print("Ieșire din modul de predicție.")
            break

        # Preprocesăm imaginea
        img_preprocessed = preprocess_image(file_path)
        # Adăugăm o dimensiune suplimentară pentru a simula un batch de imagini
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # Prezicem clasa folosind modelul antrenat
        prediction = model.predict(img_batch)
        predicted_label = np.argmax(prediction)

        print(f"Imaginea selectată este clasificată ca: {predicted_label}")


train_images = load_images('MNIST/train-images.idx3-ubyte')
train_labels = load_labels('MNIST/train-labels.idx1-ubyte')
test_images = load_images('MNIST/t10k-images.idx3-ubyte')
test_labels = load_labels('MNIST/t10k-labels.idx1-ubyte')
print("Date incarcate...")

train_images = train_images / 255.0
test_images = test_images / 255.0
print("Date normalizate...")

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(784, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
print("Modelul folosit : ", model)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                    epochs=30, batch_size=512)

train_loss, train_accuracy = model.evaluate(train_images, train_labels)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f"Eroare Antrenare: {train_loss:.4f}, Acuratete Antrenare: {train_accuracy:.4f}")
print(f"Eroare Testare: {test_loss:.4f}, Acuratete Testare: {test_accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Acuratetea de invatare')
plt.plot(history.history['val_accuracy'], label='Acuratetea de testare')
plt.xlabel('Epoca')
plt.ylabel('Acuratete')
plt.legend()
plt.title("Evolutia acuratetilor de invatare si testare")
plt.show()

plt.plot(history.history['loss'], label='Eroarea de invatare')
plt.plot(history.history['val_loss'], label='Eroarea de testare')
plt.xlabel('Epoca')
plt.ylabel('Eroare')
plt.legend()
plt.title("Evolutia erorilor de invatare si testare")
plt.show()

predictions = model.predict(test_images)
"""for i in range(3):
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f"Eticheta reala: {test_labels[i]}, Prezis: {np.argmax(predictions[i])}")
    plt.show()"""

num_images = 9
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f"Real: {test_labels[i]}, Pred: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.show()

sample_index = 524
sample_image = test_images[sample_index]
sample_label = test_labels[sample_index]

plt.imshow(sample_image, cmap="gray")
plt.title(f"Eticheta reala: {sample_label}")
plt.show()

prediction = model.predict(sample_image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)
print(f"Predictia modelului: {predicted_label}")
