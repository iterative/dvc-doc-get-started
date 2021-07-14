import tensorflow as tf
import numpy as np
import os
import time
from util import load_params, logs_to_csv

import models

MODEL_DIR = "models/fashion-mnist/"
MODEL_FILE = os.path.join(MODEL_DIR, "model.h5")
DATA_DIR = "data/fashion-mnist"
LOGS_FILE = "logs.csv"

class DVCCheckpointsCallback(tf.keras.callbacks.Callback):


    def __init__(self, model_file = None, logs_file = None, frequency = 1, append_logs = False):
        super().__init__()
        self.frequency = frequency
        self.model_file = model_file
        self.logs_file = logs_file
        self.append_logs = append_logs
        self.previous_logs = {}

    def _write_logs_to_file(self, logs):
        logs_csv = logs_to_csv(self._logs_to_list(logs))
        with open(self.logs_file, "w") as f:
            f.write(logs_csv)

    def _append_logs_to_file(self, logs):
        logs = self._logs_to_list(logs)
        if self.previous_logs == {}:
            self.previous_logs = logs
        else:
            for k in self.previous_logs:
                self.previous_logs[k] += logs[k]
        logs_csv = logs_to_csv(self.previous_logs)
        with open(self.logs_file, "w") as f:
            f.write(logs_csv)

    def _logs_to_list(self, logs):
        return {k: [v] for k, v in logs.items()}


    def dvc_signal(self):
        "Generates a DVC signal file and blocks until it's deleted"
        dvc_root = os.getenv("DVC_ROOT") # Root dir of dvc project.
        if dvc_root: # Skip if not running via dvc.
            signal_file = os.path.join(dvc_root, ".dvc", "tmp",
                "DVC_CHECKPOINT")
            with open(signal_file, "w") as f: # Write empty file.
                f.write("")
            while os.path.exists(signal_file): # Wait until dvc deletes file.
                # Wait 10 milliseconds
                time.sleep(0.01)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.model_file:
            self.model.save(self.model_file)
        if self.logs_file:
            if self.append_logs:
                self._append_logs_to_file(logs)
            else:
                self._write_logs_to_file(logs)
        if (epoch % self.frequency) == 0:
            self.dvc_signal()

def load_npz_data(filename):
    npzfile = np.load(filename)
    return (npzfile['images'], npzfile['labels'])

def history_to_csv(history):
    keys = list(history.history.keys())
    csv_string = ", ".join(["epoch"] + keys) + "\n"
    list_len = len(history.history[keys[0]])
    for i in range(list_len):
        row = str(i+1) + ", " + ", ".join([str(history.history[k][i]) for k in keys]) + "\n"
        csv_string += row

    return csv_string

def main():
    params = load_params()["train"]
    if params["resume"] and os.path.exists(MODEL_FILE):
        m = tf.keras.models.load_model(MODEL_FILE)
    else:
        m = models.get_model()
    m.summary()

    whole_train_img, whole_train_labels = load_npz_data(os.path.join(DATA_DIR,
                                                                     "preprocessed/mnist-train.npz"))
    test_img, test_labels = load_npz_data(os.path.join(DATA_DIR,
                                                       "preprocessed/mnist-test.npz"))
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

    if params["epochs"] == 0:
        callback = DVCCheckpointsCallback(logs_file=LOGS_FILE,
                                          model_file=MODEL_FILE,
                                          append_logs=True,
                                          frequency=1)
        while True:
            history = m.fit(
                x_train,
                y_train,
                batch_size=params["batch_size"],
                epochs=1,
                verbose=1,
                validation_data=(x_valid, y_valid),
                callbacks=[callback]
            )
    else:
        callback = DVCCheckpointsCallback(logs_file=LOGS_FILE,
                                          model_file=MODEL_FILE,
                                          append_logs=False,
                                          frequency=1)
        history = m.fit(
            x_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            verbose=1,
            validation_data=(x_valid, y_valid),
            callbacks=[callback]
        )


if __name__ == "__main__":
    main()
