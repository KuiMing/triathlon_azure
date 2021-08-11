import argparse
from datetime import datetime
import os
import pickle
import pandas as pd
import investpy
from sklearn.preprocessing import MinMaxScaler
from azureml.core import Run
from azureml.core import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=bool, default=False)
    parser.add_argument("--target_folder", type=str)
    parser.add_argument("--input_folder", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.history:
        if not os.path.isdir("currency"):
            os.system("mkdir currency")

        usd_twd = investpy.get_currency_cross_historical_data(
            "USD/TWD",
            from_date="01/01/1900",
            to_date=datetime.now().strftime("%d/%m/%Y"),
        )
        usd_twd.reset_index(inplace=True)
        usd_twd.to_csv("currency/usd_twd.csv", index=False)
        currency_data = usd_twd.Close.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        currency_data = scaler.fit_transform(currency_data)
        with open("currency/scaler.pickle", "wb") as f_h:
            pickle.dump(scaler, f_h)
        f_h.close()
        currency_data = usd_twd[
            (usd_twd.Date >= "2010-01-01") & (usd_twd.Date < "2021-01-01")
        ]
        currency_data.to_csv("currency/training_data.csv")
    else:
        path = os.path.join(args.target_folder, "usd_twd.csv")
        input_path = os.path.join(args.input_folder, "usd_twd.csv")
        history = pd.read_csv(input_path)

        recent = investpy.get_currency_cross_recent_data("USD/TWD")
        recent.reset_index(inplace=True)
        history = history.append(recent, ignore_index=True)
        history.drop_duplicates(subset="Date", keep="last", inplace=True)
        history.to_csv(path, index=False)
        history = history.tail(2400)
        history.to_csv(
            os.path.join(args.target_folder, "training_data.csv"), index=False
        )
        run = Run.get_context()
        work_space = run.experiment.workspace
        datastore = work_space.get_default_datastore()
        dataset = Dataset.File.from_files(path=(datastore, 'currency'))
        dataset.register(work_space, name='currency')


if __name__ == "__main__":
    main()
