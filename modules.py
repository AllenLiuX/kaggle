from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time


def date2days(date):
    d1 = datetime(2020, 4, 12)
    month = int(date[:2])
    day = int(date[3:5])
    year = int(date[6:])
    d2 = datetime(year, month, day)
    return (d2-d1).days


def days2date(num):
    d1 = datetime(2020, 4, 12)
    d2 = d1 + timedelta(days=num)
    date = d2.strftime('%m-%d-%Y')
    return date


def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def df_MAPE(true_df, pred_df):
    mape_confirmed = MAPE(true_df['Confirmed'], pred_df['Confirmed'])
    mape_deaths = MAPE(true_df['Deaths'], pred_df['Deaths'])
    return np.mean([mape_confirmed, mape_deaths])


def get_forcast_id(state, date, test_df):
    if type(date) == int:
        date = days2date(date)
    df = test_df[test_df['Province_State'] == state]
    df = df[df['Date'] == date]
    return df['ForecastID'].values[0]
    

if __name__ == '__main__':
    start_time = time.time()
    test_df = pd.read_csv('data/test.csv')
    forcast_id = get_forcast_id('California', '09-05-2020', test_df)
    print(forcast_id)
    end_time = time.time()
    print('======= Time taken: %f =======' %(end_time - start_time))
