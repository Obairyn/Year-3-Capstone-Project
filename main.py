from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

input_shape = (32, 32, 3)
num_classes = 10
batch_size = 64
epochs = 10

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                    validation_data=(test_images, test_labels))

print("Точность на тренировочных данных:", history.history['accuracy'][-1])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

predictions = model.predict(test_images)

test_labels = np.array(test_labels)

correct_indices = np.nonzero(np.argmax(predictions, axis=1) == np.squeeze(test_labels))

correct_images = test_images[correct_indices]
correct_labels = np.squeeze(test_labels[correct_indices])

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(correct_images[i])
    plt.title(class_names[correct_labels[i]])
    plt.axis('off')
plt.show()
