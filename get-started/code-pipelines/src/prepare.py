import struct
import numpy as np
import gzip
import os
from util import load_params

def mnist_images_idx_to_array(images_filename):
    images_f = gzip.open(images_filename, mode="rb")
    images_f.seek(0)
    magic = struct.unpack('>I', images_f.read(4))[0]
    if magic != 0x00000803:
        raise Exception(f"Format error: Need an IDX3 file: {images_filename}")
    n_images = struct.unpack('>I', images_f.read(4))[0]
    n_row = struct.unpack('>I', images_f.read(4))[0]
    n_col = struct.unpack('>I', images_f.read(4))[0]

    n_bytes = n_images * n_row * n_col # each pixel is 1 byte

    images_data = struct.unpack('>' + str(n_bytes) + 'B', images_f.read(n_bytes))

    images_array = np.asarray(images_data, dtype='uint8')
    images_array.shape = (n_images, n_row, n_col)

    return images_array

def mnist_labels_idx_to_array(labels_filename):
    labels_f = gzip.open(labels_filename, mode="rb")
    labels_f.seek(0)
    magic = struct.unpack('>I', labels_f.read(4))[0]
    if magic != 0x00000801:
        raise Exception(f"Format error: Need an IDX file: {labels_filename}")
    n_labels = struct.unpack('>I', labels_f.read(4))[0]
    labels_data = struct.unpack('>' + str(n_labels) + 'B', labels_f.read(n_labels))
    labels_array = np.asarray(labels_data, dtype='uint8')
    return labels_array

def remix(images1, images2, labels1, labels2, seed, split):
    images = np.vstack((images1, images2))
    labels = np.hstack((labels1, labels2))

    assert(images.shape[0] == labels.shape[0])

    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    split_index = int(images.shape[0] * (1 - split))

    images1 = images[:split_index]
    labels1 = labels[:split_index]
    images2 = images[split_index:]
    labels2 = labels[split_index:]

    return (images1, images2, labels1, labels2)


def main():
    params = load_params()["prepare"]
    print(params)

    training_images = mnist_images_idx_to_array("data/mnist/raw/train-images-idx3-ubyte.gz")
    # print(f"Read training data: {training_images}")
    training_labels = mnist_labels_idx_to_array("data/mnist/raw/train-labels-idx1-ubyte.gz")
    # print(f"Read training labels: {training_labels}")
    testing_images = mnist_images_idx_to_array("data/mnist/raw/t10k-images-idx3-ubyte.gz")
    # print(f"Read testing data: {testing_images}")
    testing_labels = mnist_labels_idx_to_array("data/mnist/raw/t10k-labels-idx1-ubyte.gz")
    # print(f"Read testing labels: {testing_labels}")

    if params["remix"]:
        training_images, testing_images, training_labels, testing_labels = remix(images1=training_images,
                                                                                 images2=testing_images,
                                                                                 labels1=training_labels,
                                                                                 labels2=testing_labels,
                                                                                 seed=params["seed"],
                                                                                 split=params["remix_split"])

        assert training_images.shape[0] + testing_images.shape[0] == 70000
        assert training_labels.shape[0] + testing_labels.shape[0] == 70000

    print(f"Training Dataset Shape: {training_images.shape}")
    print(f"Testing Dataset Shape: {testing_images.shape}")
    print(f"Training Labels: {training_labels}")
    print(f"Testing Labels: {testing_labels}")

    os.makedirs("data/mnist/prepared")

    np.savez("data/mnist/prepared/mnist-train.npz", images=training_images, labels=training_labels)
    np.savez("data/mnist/prepared/mnist-test.npz", images=testing_images, labels=testing_labels)


if __name__ == "__main__":
    main()
