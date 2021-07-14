import tensorflow as tf
from util import load_params

def mlp(dense_units=128, activation="relu"):
    return tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(dense_units, activation=activation),
      tf.keras.layers.Dense(10, activation="softmax")
])

def cnn(dense_units=128, conv_kernel=(3,3), conv_units=32, dropout=0.5, activation="relu"):
    return tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape=(28, 28),
                                target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(conv_units,
                               kernel_size=conv_kernel,
                               activation=activation),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation=activation),
        tf.keras.layers.Dense(10, activation="softmax")])


def get_model():
    model_params = load_params()["model"]

    if model_params["name"].lower() == "mlp":
        p = model_params["mlp"]
        model = mlp(p["units"], p["activation"])
    elif model_params["name"].lower() == "cnn":
        p = model_params["cnn"]
        model = cnn(dense_units=p["dense_units"],
                    conv_kernel=(p["conv_kernel_size"], p["conv_kernel_size"]),
                    conv_units=p["conv_units"],
                    dropout=p["dropout"],
                    activation=p["activation"])
    else:
        raise Exception(f"No Model with the name {model_params['name']} is defined")

    if model_params["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam()
    elif model_params["optimizer"].lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD()
    elif model_params["optimizer"].lower() == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop()
    elif model_params["optimizer"].lower() == "adadelta":
        optimizer = tf.keras.optimizers.Adadelta()
    elif model_params["optimizer"].lower() == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad()
    elif model_params["optimizer"].lower() == "adamax":
        optimizer = tf.keras.optimizers.Adamax()
    elif model_params["optimizer"].lower() == "nadam":
        optimizer = tf.keras.optimizers.Nadam()
    elif model_params["optimizer"].lower() == "ftrl":
        optimizer = tf.keras.optimizers.Ftrl()
    else:
        raise Exception(f"No optimizer with the name {model_params['optimizer']} is defined")

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(curve="ROC", name="ROC", multi_label=True),
        tf.keras.metrics.AUC(curve="PR", name="PR", multi_label=True),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives()]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


