from datetime import datetime
import pickle
import investpy
from sklearn.preprocessing import MinMaxScaler


def main():
    usd_twd = investpy.get_currency_cross_historical_data(
        "USD/TWD", from_date="01/01/1900", to_date=datetime.now().strftime("%d/%m/%Y")
    )
    usd_twd.reset_index(inplace=True)
    usd_twd.to_csv("currency/usd_twd.csv", index=False)
    currency_data = usd_twd.Close.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    currency_data = scaler.fit_transform(currency_data)
    with open("currency/usd_twd.pickle", "wb") as f_h:
        pickle.dump(scaler, f_h)
    f_h.close()
    currency_data = usd_twd[
        (usd_twd.Date >= "2010-01-01") & (usd_twd.Date < "2021-01-01")
    ]
    currency_data.to_csv("currency/traing_data.csv")


if __name__ == "__main__":
    main()
