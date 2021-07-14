import tensorflow as tf
import numpy as np
import os

from util import load_params, history_to_csv, history_list_to_csv
import models
from dvc.api import make_checkpoint
import dvclive

MODEL_DIR = "models/fashion-mnist/"
MODEL_FILE = os.path.join(MODEL_DIR, "model.h5")
DATA_DIR = "data/fashion-mnist"

class DVCLiveCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_file = None):
        super().__init__()
        self.model_file = model_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.model_file:
            self.model.save(self.model_file)
        for metric, value in logs.items():
            dvclive.log(metric, value)
        dvclive.next_step()


def load_npz_data(filename):
    npzfile = np.load(filename)
    return (npzfile['images'], npzfile['labels'])

def main():
    params = load_params()["train"]
    if params["resume"] and os.path.exists(MODEL_FILE):
        m = tf.keras.models.load_model(MODEL_FILE)
    else:
        m = models.get_model()
    m.summary()

    whole_train_img, whole_train_labels = load_npz_data(os.path.join(DATA_DIR, "preprocessed/mnist-train.npz"))
    test_img, test_labels = load_npz_data(os.path.join(DATA_DIR, "preprocessed/mnist-test.npz"))
    validation_split_index = int((1 - params["validation_split"]) * whole_train_img.shape[0])
    if validation_split_index == whole_train_img.shape[0]:
        x_train = whole_train_img
        x_valid = test_img
        y_train = whole_train_labels
        y_valid = test_labels
    else:
        x_train = whole_train_img[:validation_split_index]
        x_valid = whole_train_img[validation_split_index:]
        y_train = whole_train_labels[:validation_split_index]
        y_valid = whole_train_labels[validation_split_index:]

    print(f"x_train: {x_train.shape}")
    print(f"x_valid: {x_valid.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_valid: {y_valid.shape}")

    dvclive.init("training_metrics")

    if params["epochs"] == 0:
        history_list = []
        while True:
            history = m.fit(
                x_train,
                y_train,
                batch_size=params["batch_size"],
                epochs=1,
                verbose=1,
                validation_data=(x_valid, y_valid),
                callbacks=[DVCLiveCallback(model_file=MODEL_FILE)]
            )
    else:
        history = m.fit(
            x_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            verbose=1,
            validation_data=(x_valid, y_valid),
            callbacks=[DVCLiveCallback(model_file=MODEL_FILE)]
        )

if __name__ == "__main__":
    main()
