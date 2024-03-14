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

# predictions = model.predict(test_images)

# correct_images = []
# incorrect_images = []

# for i in range(len(test_labels)):
#     predicted_label = np.argmax(predictions[i])
#     true_label = test_labels[i]
#     if predicted_label == true_label:
#         correct_images.append((test_images[i], predicted_label))
#     else:
#         incorrect_images.append((test_images[i], predicted_label, true_label))

# plt.figure(figsize=(10, 10))
# plt.suptitle("Правильно классифицированные изображения", fontsize=16)
# for i in range(2):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(correct_images[i][0], cmap=plt.cm.binary)
#     plt.title("Предсказано: {}".format(class_names[int(correct_images[i][1])])) 
#     plt.axis('off')

# plt.figure(figsize=(10, 10))
# plt.suptitle("Неправильно классифицированные изображения", fontsize=16)
# for i in range(2):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(incorrect_images[i][0], cmap=plt.cm.binary)
#     plt.title("Предсказано: {}, На самом деле: {}".format(
#         class_names[incorrect_images[i][1]], class_names[incorrect_images[i][2]]))
#     plt.axis('off')

# plt.show()
predictions = model.predict(test_images)

correct_images = []
incorrect_images = []

for i in range(len(test_labels)):
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        correct_images.append((test_images[i], predicted_label))
    else:
        incorrect_images.append((test_images[i], predicted_label, true_label))

plt.figure(figsize=(10, 10))
plt.suptitle("Правильно классифицированные изображения", fontsize=16)
for i in range(2):
    plt.subplot(2, 2, i + 1)
    plt.imshow(correct_images[i][0], cmap=plt.cm.binary)
    try:
        predicted_class_name = class_names[correct_images[i][1]]
    except IndexError:
        predicted_class_name = "Unknown"
    plt.title(f"Predicted: {predicted_class_name} (Correct)")
    plt.axis('off')

plt.figure(figsize=(10, 10))
plt.suptitle("Неправильно классифицированные изображения", fontsize=16)
for i in range(2):
    plt.subplot(2, 2, i + 1)
    plt.imshow(incorrect_images[i][0], cmap=plt.cm.binary)
    plt.title(f"Predicted: {class_names[incorrect_images[i][1]]}, Actual: {class_names[incorrect_images[i][2]]} (Incorrect)")
    plt.axis('off')

plt.show()