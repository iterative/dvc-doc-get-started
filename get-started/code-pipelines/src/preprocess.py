import tensorflow as tf
import numpy as np
import os
from util import load_params


def normalize(images_array):
    return images_array / 255

def add_noise(images, s_vs_p=0.5, amount=0.0004):
    n_images, n_row, n_col = images.shape
    out = np.copy(images)
    # Salt mode
    n_salt = np.ceil(amount * images.size * s_vs_p)
    salt_coords = [np.random.randint(0, i - 1, int(n_salt))
            for i in image.shape]
    out[salt_coords] = 1

    # Pepper mode
    n_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    pepper_coords = [np.random.randint(0, i - 1, int(n_pepper))
            for i in image.shape]
    out[pepper_coords] = 0

    return out

def load_npz_data(filename):
    npzfile = np.load(filename)
    return (npzfile['images'], npzfile['labels'])

def shuffle_in_parallel(seed, array1, array2):
    np.random.seed(seed)
    np.random.shuffle(array1)
    np.random.seed(seed)
    np.random.shuffle(array2)

    return array1, array2

def main():
    params = load_params()["preprocess"]
    print(params)

    training_images, training_labels = load_npz_data("data/mnist/prepared/mnist-train.npz")
    testing_images, testing_labels = load_npz_data("data/mnist/prepared/mnist-test.npz")

    seed = params["seed"]

    if params["normalize"]:
        training_images = normalize(training_images)
        testing_images = normalize(testing_images)

    if params["shuffle"]:
        training_images, training_labels = shuffle_in_parallel(seed, training_images, training_labels)
        testing_images, testing_labels = shuffle_in_parallel(seed, testing_images, testing_labels)

    training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=10, dtype="float32")
    testing_labels = tf.keras.utils.to_categorical(testing_labels, num_classes = 10, dtype="float32")

    print(f"Training Images: {training_images.shape} - {training_images.dtype}")
    print(f"Testing Images: {testing_images.shape} - {testing_images.dtype}")
    
    if not os.path.exists("data/mnist/preprocessed"):
        os.makedirs("data/mnist/preprocessed")
    np.savez("data/mnist/preprocessed/mnist-train.npz", images=training_images, labels=training_labels)
    np.savez("data/mnist/preprocessed/mnist-test.npz", images=testing_images, labels=testing_labels)

if __name__ == "__main__":
    main()
