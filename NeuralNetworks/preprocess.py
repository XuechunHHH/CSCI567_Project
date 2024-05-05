import pandas as pd

def preprocess_data():
    df_train, df_val, df_test = pd.read_csv("train.csv"), pd.read_csv("val.csv"), pd.read_csv("test.csv")
    
    training_data, training_label = get_np_data(df_train)
    validation_data, validation_label = get_np_data(df_val)
    testing_data, testing_label = get_np_data(df_test)

    return training_data, training_label, validation_data, validation_label, testing_data, testing_label

def get_np_data(df):
    df_data = df.drop(['ID', 'Status'], axis=1)
    df_label = df['Status']

    return df_data.to_numpy(), df_label.to_numpy()