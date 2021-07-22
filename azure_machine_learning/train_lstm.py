import argparse
import os
import pickle
import numpy as np
from azureml.core.run import Run
from azureml.core.model import Model
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import TensorBoard, EarlyStopping


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
    x_train = x_all[: int(len(x_all) * (1 - rate))]
    y_train = y_all[: int(y_all.shape[0] * (1 - rate))]
    x_val = x_all[int(len(x_all) * (1 - rate)) :]
    y_val = y_all[int(y_all.shape[0] * (1 - rate)) :]
    return x_train, y_train, x_val, y_val


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, help="Path to the training data")
    parser.add_argument(
        "--experiment",
        type=bool,
        default=False,
        help="Just run an experiment, there is no pipeline",
    )
    parser.add_argument(
        "--log_folder", type=str, help="Path to the log", default="./logs"
    )
    args = parser.parse_args()
    return args


def load_best_model(work_space, model_name="currency", key="val_loss"):
    model_obj = Model(work_space, model_name)
    model_list = model_obj.list(work_space, name=model_name)
    values = []
    version = []
    for i in model_list:
        values.append(float(i.properties[key]))
        version.append(i.version)
    model_obj = Model(work_space, model_name, version=version[np.argmin(values)])
    model_name = model_obj.download(exist_ok=True)
    model = load_model(model_name)
    return model


def main():
    """
    Training of LeNet with keras
    """
    args = parse_args()
    run = Run.get_context()
    # Load mnist data
    usd_twd = pd.read_csv(os.path.join(args.target_folder, "training_data.csv"))
    data = usd_twd.Close.values.reshape(-1, 1)
    with open(os.path.join(args.target_folder, "scaler.pickle"), "rb") as f_h:
        scaler = pickle.load(f_h)
    f_h.close()
    data = scaler.fit_transform(data)
    data_len = 240
    x_train, y_train, x_val, y_val = data_generator(data, data_len)

    if args.experiment:
        model = Sequential()
        model.add(LSTM(16, input_shape=(data_len, 1)))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        # Tensorboard
        callback = TensorBoard(
            log_dir=args.log_folder,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
        )
    else:
        work_space = run.experiment.workspace
        model = load_best_model(work_space, model_name="currency", key="val_loss")
        print("Load Model")
        callback = EarlyStopping(
            monitor="val_loss", mode="min", min_delta=1e-8, patience=50
        )
    # train the network
    history_callback = model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=240,
        verbose=1,
        validation_data=[x_val, y_val],
        callbacks=[callback],
    )

    # ouput log
    metrics = history_callback.history
    run.log_list("train_loss", metrics["loss"][:10])
    run.log_list("val_loss", metrics["val_loss"][:10])
    run.log_list("start", [usd_twd.Date.values[0]])
    run.log_list("end", [usd_twd.Date.values[-1]])
    run.log_list("epoch", [len(history_callback.epoch)])

    print("Finished Training")
    model.save("outputs/keras_lstm.h5")
    print("Saved Model")
    if args.experiment:
        with open("outputs/scaler.pickle", "wb") as f_h:
            pickle.dump(scaler, f_h)
        f_h.close()
    else:
        model = Model.register(
            workspace=work_space,
            model_name="currency",
            tags={"model": "LSTM"},
            model_path="outputs/keras_lstm.h5",
            model_framework="keras",
            model_framework_version="2.2.4",
            properties={
                "train_loss": metrics["loss"][-1],
                "val_loss": metrics["val_loss"][-1],
                "data": "USD/TWD from {0} to {1}".format(
                    usd_twd.Date.values[0], usd_twd.Date.values[-1]
                ),
                "epoch": len(history_callback.epoch),
            },
        )
        print("Registered Model")


if __name__ == "__main__":
    main()
