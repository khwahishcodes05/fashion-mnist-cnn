# ==========================================
# Fashion MNIST Image Classification using CNN
# ==========================================

# Import Required Libraries
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# Load Fashion MNIST Dataset
# ==========================================
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# ==========================================
# Reshape Images
# ==========================================
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ==========================================
# Normalize Images
# ==========================================
X_train = X_train / 255.0
X_test = X_test / 255.0

# ==========================================
# Build CNN Model
# ==========================================
model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),

    Dropout(0.3),

    Dense(128, activation="relu"),
    Dense(64, activation="relu"),

    Dense(10, activation="softmax")
])

# ==========================================
# Compile the Model
# ==========================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# Early Stopping
# ==========================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==========================================
# Train the Model
# ==========================================
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# Evaluate the Model
# ==========================================
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# ==========================================
# Plot Accuracy Graph
# ==========================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# ==========================================
# Plot Loss Graph
# ==========================================
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ==========================================
# Class Names
# ==========================================
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# ==========================================
# Predict on Test Images
# ==========================================
predictions = model.predict(X_test)

predicted_labels = predictions.argmax(axis=1)

print("Predicted:", class_names[predicted_labels[0]])
print("Actual   :", class_names[y_test[0]])

# ==========================================
# Display First Test Image with Prediction
# ==========================================
plt.figure(figsize=(4,4))
plt.imshow(X_test[0].reshape(28,28), cmap="gray")
plt.title(f"Predicted: {class_names[predicted_labels[0]]}\nActual: {class_names[y_test[0]]}")
plt.axis("off")
plt.show()