import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Input
import keras.models as M

def load_process_data():
    feature_cols = ["retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "parks_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline",
            "residential_percent_change_from_baseline"]
    mob_df = pd.read_csv('new_mobility.csv')
    policy_df = pd.read_csv('new_policy.csv')
    mob_df["mobility_index"] = mob_df[feature_cols].mean(axis=1)
    mob_df = mob_df.drop(feature_cols, axis=1)

    policy_df = policy_df.drop(["stringency_index","deaths"], axis=1)

    policy_df["confirmed_cases"] = (policy_df.groupby(["country_name"])['confirmed_cases'].apply(pd.Series.pct_change) + 1)

    agg = mob_df.merge(policy_df, how="left", left_on=["country_region", "date"], right_on=["country_name", "date"])
    

    
    processed_data = agg.replace([np.inf, -np.inf], 1)
    processed_data = processed_data.drop(["Unnamed: 0", "country_region"],axis=1)

    label = []
    data = []
    for _, df in processed_data.groupby("country_name"):
        if df.confirmed_cases.count() >= 209:
            df = df.drop(["country_name"],axis=1)
            df = df.sort_values("date")
            label.append(df.pop("mobility_index").to_numpy())
            data.append(df.to_numpy())
    
    return data, label


def lstm_model():
    data, label = load_process_data()
    model_input = Input(shape=(data[0].shape))

    model_output = LSTM(4, return_sequences=True)(model_input)

    model = M.Model(model_input, model_output)
    model.compile('sgd', 'mean_squared_error')

    model.fit(data, label, epochs=400, verbose=1)
    model.save('lstm_model.h5')
    return model

if __name__ == "__main__":
    lstm_model()
