import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images (28x28 → 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# neural network model
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    
    tf.keras.layers.Dense(64, activation='relu'),
    
    tf.keras.layers.Dense(10, activation='softmax')
])

#using Adam optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Training the model
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_test, y_test)
)

os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Plot training & validation accuracy
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.savefig("plots/adam_accuracy_curve.png")
plt.show()

# Plot training & validation loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig("plots/adam_loss_curve.png")
plt.show()

# Evaluate original loss
original_loss, original_acc = model.evaluate(x_test, y_test, verbose=0)

print("Original Test Loss:", original_loss)

sharpness_scores = []

for i in range(10):   # repeat perturbation 10 times
    original_weights = model.get_weights()
    noisy_weights = []
    for w in original_weights:
        noise = np.random.normal(0, 0.01, w.shape)
        noisy_weights.append(w + noise)
    model.set_weights(noisy_weights)
    
    # evaluate loss
    noisy_loss, _ = model.evaluate(x_test, y_test, verbose=0)
    
    # compute sharpness
    sharpness = (noisy_loss - original_loss) / original_loss
    sharpness_scores.append(sharpness)
    model.set_weights(original_weights)

avg_sharpness = np.mean(sharpness_scores)
std_sharpness = np.std(sharpness_scores)

print(f"Average Sharpness Score: {avg_sharpness:.5f} ± {std_sharpness:.5f}")

# Plot sharpness distribution
plt.figure(figsize=(8,5))

plt.hist(sharpness_scores, bins=10, color='purple', edgecolor='black')

plt.xlabel("Sharpness Score")
plt.ylabel("Frequency")
plt.title("Sharpness Distribution for Adam Minimum")

plt.savefig("plots/adam_sharpness_distribution.png")

plt.show()

# Loss vs Noise Magnitude Experiment
noise_levels = np.linspace(0, 0.01, 6)
loss_values = []

original_weights = model.get_weights()

for noise_level in noise_levels:

    noisy_weights = []

    for w in original_weights:
        noise = np.random.normal(0, noise_level, w.shape)
        noisy_weights.append(w + noise)

    model.set_weights(noisy_weights)

    loss, _ = model.evaluate(x_test, y_test, verbose=0)

    loss_values.append(loss)

model.set_weights(original_weights)

# Plot loss vs noise magnitude
plt.figure(figsize=(8,5))

plt.plot(noise_levels, loss_values, marker='o')

plt.xlabel("Noise Magnitude")
plt.ylabel("Loss")
plt.title("Loss vs Weight Perturbation Magnitude")

plt.savefig("plots/adam_loss_vs_noise_curve.png")

plt.show()

with open("results/adam_results.txt", "w") as f:
    f.write(f"Original Test Loss: {original_loss}\n")
    f.write(f"Average Sharpness Score: {avg_sharpness:.5f} ± {std_sharpness:.5f}\n")

# 2D LOSS LANDSCAPE CONTOUR PLOT
original_weights = model.get_weights()

# Generate two random directions
direction1 = [np.random.normal(size=w.shape) for w in original_weights]
direction2 = [np.random.normal(size=w.shape) for w in original_weights]

# Normalize directions (important!)
direction1 = [d / (np.linalg.norm(d) + 1e-10) for d in direction1]
direction2 = [d / (np.linalg.norm(d) + 1e-10) for d in direction2]

# Define grid range
alphas = np.linspace(-0.05, 0.05, 20)
betas = np.linspace(-0.05, 0.05, 20)

loss_grid = np.zeros((len(alphas), len(betas)))

# Compute loss over grid
for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):

        new_weights = []

        for w, d1, d2 in zip(original_weights, direction1, direction2):
            new_w = w + alpha * d1 + beta * d2
            new_weights.append(new_w)

        model.set_weights(new_weights)

        loss, _ = model.evaluate(x_test, y_test, verbose=0)
        loss_grid[i, j] = loss

# Restore original weights
model.set_weights(original_weights)

# Plot contour
plt.figure(figsize=(7,6))

X, Y = np.meshgrid(alphas, betas)

contour = plt.contourf(X, Y, loss_grid.T, levels=20)

plt.colorbar(contour)

plt.xlabel("Direction 1 (α)")
plt.ylabel("Direction 2 (β)")
plt.title("2D Loss Landscape Contour around Adam Minimum")

plt.savefig("plots/adam_contour.png")

plt.show()
