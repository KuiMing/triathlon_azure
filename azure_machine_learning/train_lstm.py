import argparse
import os
import pickle
import numpy as np
from azureml.core import Run
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import TensorBoard


# usd_twd = investpy.get_currency_cross_historical_data(
#     "USD/TWD", from_date="01/01/1900", to_date="31/12/2020"
# )
# usd_twd.reset_index(inplace=True)


def data_generator(data, data_len=240):
    generator = TimeseriesGenerator(
        data=data, targets=range(data.shape[0]), length=data_len, batch_size=1, stride=1
    )
    x_all = []
    for i in generator:
        x_all.append(i[0][0])
    x_all = np.array(x_all)
    y_all = data[range(data_len, len(x_all) + data_len)]
    rate = 0.4
    x_test = x_all[-int(len(x_all) * rate) :]
    y_test = y_all[-int(y_all.shape[0] * rate) :]
    x_train = x_all[: int(len(x_all) * (1 - rate) ** 2)]
    y_train = y_all[: int(y_all.shape[0] * (1 - rate) ** 2)]
    x_val = x_all[int(len(x_all) * (1 - rate) ** 2) : int(len(x_all) * (1 - rate))]
    y_val = y_all[
        int(y_all.shape[0] * (1 - rate) ** 2) : int(y_all.shape[0] * (1 - rate))
    ]
    return x_train, y_train, x_val, y_val, x_test, y_test


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, help="Path to the training data")
    parser.add_argument(
        "--log_folder", type=str, help="Path to the log", default="./logs"
    )
    args = parser.parse_args()
    return args


def main():
    """
    Training of LeNet with keras
    """
    args = parse_args()
    print("===== DATA =====")
    print("DATA PATH: {}".format(args.data_folder))
    print("LIST FILES IN DATA PATH...")
    print("================")
    run = Run.get_context()

    # Load mnist data
    usd_twd = pd.read_csv(os.path.join(args.data_folder, "usd_twd.csv"))
    data = usd_twd.Close.values.reshape(-1, 1)
    with open(os.path.join(args.data_folder, "scaler.pickle"), "rb") as f_h:
        scaler = pickle.load(f_h)
    f_h.close()
    data = scaler.fit_transform(data)
    data = data[usd_twd[usd_twd.Date >= "2010-01-01"].index]
    data_len = 240
    x_train, y_train, x_val, y_val, _, _ = data_generator(data, data_len)
    model = Sequential()
    model.add(LSTM(16, input_shape=(data_len, 1)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    # Tensorboard
    tb_callback = TensorBoard(
        log_dir=args.log_folder,
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
    )

    # train the network
    history_callback = model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=240,
        verbose=1,
        validation_data=[x_val, y_val],
        callbacks=[tb_callback],
    )

    # ouput log
    metrics = history_callback.history
    run.log_list("train_loss", metrics["loss"])
    run.log_list("train_accuracy", metrics["accuracy"])
    run.log_list("val_loss", metrics["val_loss"])
    run.log_list("val_accuracy", metrics["val_accuracy"])

    run.register_model(
        model_name=args.experiment_name,
        tags={"data": "mnist", "model": "classification"},
        model_path="outputs/keras_lenet.h5",
        model_framework="keras",
        model_framework_version="2.3.1",
        properties={
            "train_loss": metrics["train_loss"][-1],
            "train_accuracy": metrics["train_accuracy"][-1],
            "val_loss": metrics["val_loss"][-1],
            "val_accuracy": metrics["val_accuracy"][-1],
        },
    )
    print("Finished Training")
    model.save("outputs/keras_lenet.h5")
    print("Saved Model")


if __name__ == "__main__":
    main()
