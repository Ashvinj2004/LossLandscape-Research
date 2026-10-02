import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.datasets import cifar10

# ==============================
# LOAD DATA
# ==============================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images
x_train = x_train.reshape(-1, 32*32*3)
x_test = x_test.reshape(-1, 32*32*3)

# Fix labels
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# ==============================
# MODEL
# ==============================
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(3072,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# ==============================
# COMPILE (FIXED LOSS)
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # ✅ FIXED
    metrics=['accuracy']
)

model.summary()

# ==============================
# TRAINING
# ==============================
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# ==============================
# FOLDERS
# ==============================
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==============================
# ACCURACY PLOT
# ==============================
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CIFAR-10 SGD: Training vs Validation Accuracy')
plt.legend()
plt.savefig("plots/cifar_sgd_accuracy.png")
plt.close()

# ==============================
# LOSS PLOT
# ==============================
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CIFAR-10 SGD: Training vs Validation Loss')
plt.legend()
plt.savefig("plots/cifar_sgd_loss.png")
plt.close()

# ==============================
# EVALUATION
# ==============================
original_loss, original_acc = model.evaluate(x_test, y_test, verbose=0)

print("Original Test Loss:", original_loss)
print("Test Accuracy:", original_acc)

# ==============================
# SHARPNESS
# ==============================
sharpness_scores = []

for i in range(10):
    original_weights = model.get_weights()
    noisy_weights = []

    for w in original_weights:
        noise = np.random.normal(0, 0.01, w.shape)
        noisy_weights.append(w + noise)

    model.set_weights(noisy_weights)

    noisy_loss, _ = model.evaluate(x_test, y_test, verbose=0)

    sharpness = (noisy_loss - original_loss) / original_loss
    sharpness_scores.append(sharpness)

    model.set_weights(original_weights)

avg_sharpness = np.mean(sharpness_scores)
std_sharpness = np.std(sharpness_scores)

print(f"Average Sharpness Score: {avg_sharpness:.5f} ± {std_sharpness:.5f}")

# ==============================
# SHARPNESS PLOT
# ==============================
plt.figure()
plt.hist(sharpness_scores, bins=10)
plt.xlabel("Sharpness Score")
plt.ylabel("Frequency")
plt.title("CIFAR-10 SGD: Sharpness Distribution")
plt.savefig("plots/cifar_sgd_sharpness.png")
plt.close()

# ==============================
# LOSS VS NOISE
# ==============================
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

plt.figure()
plt.plot(noise_levels, loss_values, marker='o')
plt.xlabel("Noise Magnitude")
plt.ylabel("Loss")
plt.title("CIFAR-10 SGD: Loss vs Perturbation")
plt.savefig("plots/cifar_sgd_loss_vs_noise.png")
plt.close()

# ==============================
# SAVE RESULTS
# ==============================
with open("results/cifar_sgd_results.txt", "w") as f:
    f.write(f"Test Accuracy: {original_acc}\n")
    f.write(f"Original Test Loss: {original_loss}\n")
    f.write(f"Average Sharpness Score: {avg_sharpness:.5f} ± {std_sharpness:.5f}\n")

# ==============================
# 2D LOSS LANDSCAPE
# ==============================
original_weights = model.get_weights()

direction1 = [np.random.normal(size=w.shape) for w in original_weights]
direction2 = [np.random.normal(size=w.shape) for w in original_weights]

direction1 = [d / (np.linalg.norm(d) + 1e-10) for d in direction1]
direction2 = [d / (np.linalg.norm(d) + 1e-10) for d in direction2]

alphas = np.linspace(-0.05, 0.05, 20)
betas = np.linspace(-0.05, 0.05, 20)

loss_grid = np.zeros((len(alphas), len(betas)))

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):

        new_weights = []

        for w, d1, d2 in zip(original_weights, direction1, direction2):
            new_w = w + alpha * d1 + beta * d2
            new_weights.append(new_w)

        model.set_weights(new_weights)

        loss, _ = model.evaluate(x_test, y_test, verbose=0)
        loss_grid[i, j] = loss

model.set_weights(original_weights)

plt.figure()
X, Y = np.meshgrid(alphas, betas)

contour = plt.contourf(X, Y, loss_grid.T, levels=20)
plt.colorbar(contour)

plt.xlabel("Direction 1 (α)")
plt.ylabel("Direction 2 (β)")
plt.title("CIFAR-10 SGD: Loss Landscape")

plt.savefig("plots/cifar_sgd_contour.png")
plt.close()