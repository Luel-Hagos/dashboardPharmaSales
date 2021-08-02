import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import bisect

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


from datetime import datetime
import mlflow

import logging
import pickle

st.set_page_config('Dashboard', layout="wide")
st.title('Pharmaceutical Sales prediction across multiple stores') 
store_data = pd.read_csv('./data/store.csv', na_values=['?', None, 'undefined'])
train_data = pd.read_csv('./data/train.csv', na_values=['?', None, 'undefined'])
test_data = pd.read_csv('./data/test.csv', na_values=['?', None, 'undefined'])
submission_data = pd.read_csv('./data/sample_submission.csv', na_values=['?', None, 'undefined'])
#print(data.head(5))
st.write("Train data")
st.write(train_data)

st.write("Test data")
st.write(test_data)

st.write("Store data")
st.write(store_data)

#find the date part of the DatetimeIndex object.
def holiday(x):
    if x in ['a','b','c']:
        return 1
    return 0
    
def day_month_year(df, col):
    try:
        df['Day'] = pd.DatetimeIndex(df[col]).day
        df['Month'] = pd.DatetimeIndex(df[col]).month
        df['Year'] = pd.DatetimeIndex(df[col]).year
    except KeyError:
        print("Unknown Column Index")
    
train_data['Holiday'] = train_data['StateHoliday'].apply(holiday)
test_data['Holiday'] = test_data['StateHoliday'].apply(holiday)
train_data['Holiday'] = train_data['Holiday'] | train_data['SchoolHoliday']
test_data['Holiday'] = test_data['Holiday'] | test_data['SchoolHoliday']
day_month_year(train_data, 'Date') 
day_month_year(test_data, 'Date')

# drop missing value
test_data = test_data.dropna()
treain_data = train_data.dropna()

def weekends(x):
    if x >= 6:
        return 1
    return 0

def time_of_month(x):
    if x <= 10:
        return 0
    if x <= 20:
        return 1
    return 2

def label_holidays(x):
    if x in [0,'0','a','b','c']:
        return [0,'0','a','b','c'].index(x)
    return 5

def days_from_holiday(dates, holidays):
    days_till, days_after = [], []
    for day in dates:
        ind = bisect.bisect(holidays, day)
        if ind == 0:
            days_till.append((holidays[ind] - day).days)
            days_after.append(14)
        elif ind == len(holidays):
            days_till.append(14)
            days_after.append((day - holidays[ind - 1]).days)
        else:
            days_till.append((day - holidays[ind - 1]).days)
            days_after.append((holidays[ind] - day).days)
    return days_till, days_after

# change data formate 
train_data['Weekend'] = train_data['DayOfWeek'].apply(weekends)
test_data['Weekend'] = test_data['DayOfWeek'].apply(weekends)
train_data['TimeOfMonth'] = train_data['Day'].apply(time_of_month)
test_data['TimeOfMonth'] = test_data['Day'].apply(time_of_month)
train_data['Holiday'] = train_data['StateHoliday'].apply(label_holidays)
test_data['Holiday'] = test_data['StateHoliday'].apply(label_holidays)
train_data['Date'] = pd.DatetimeIndex(train_data['Date'])
test_data['Date'] = pd.DatetimeIndex(test_data['Date'])

def desciptin(dcs):
    if dcs == 'Train data description':
        st.write('Train data description after preprocessing') 
        st.write(train_data.describe())
    elif dcs == 'Test data description':
        st.write('Test data description after preprocessing') 
        st.write(test_data.describe())

st.title('Data description after preprocessing')
d = st.selectbox('select data description', ('Train data description', 'Test data description'))
desciptin(d)

def cor(c):
    if c == 'sales and number of customers':
        # Correlation between Sales and number of customers
        t = train_data[['Customers','Sales']].corr()
        st.write(t)
    elif c == 'sales and number of customers':
        # Correlation between promo and sales
        t = train_data[['Promo','Sales']].corr()
        st.write(t)
    elif c == 'promo, customers and sales':
        # Correlation between promo, Customers and sales
        t = train_data[['Promo', 'Customers','Sales']].corr()
        st.write(t)

st.title('Analysis')
c = st.selectbox('select correlation', ('promo and sales', 'promo, customers and sales', 'sales and number of customers'))
cor(c)



# def total_data_description(data, t):
#     if t == 'Email':
#         st.write('Description of total data volume (in Bytes) for Email ') 
#         st.write(data['Email'].describe())
#     elif t == 'Social media':
#         st.write('Description of total data volume (in Bytes) for Socal media')
#         st.write(data['Social Media'].describe())
#     elif t == 'Google':
#         st.write('Description of total data volume (in Bytes) for Google')
#         st.write(data['Google'].describe())
#     elif t == 'Youtube':
#         st.write('Description of total data volume (in Bytes) for Youtube')
#         st.write(data['Youtube'].describe())
#     elif t == 'Netflix':
#         st.write('Description of total data volume (in Bytes) for Netflix')
#         st.write(data['Netflix'].describe())
#     elif t == 'Gaming':
#         st.write('Description of total data volume (in Bytes) for Gaming')
#         st.write(data['Gaming'].describe())
#     else:
#         st.write(data)

# def user_engagement_analysis(data, a):
#     if a == 'sessions frequency' :
#         # top 10 sessions frequency
#         st.write('top 5 sessions frequency')
#         sessions_frequency = data.groupby('MSISDN/Number')
#         sessions_frequency = sessions_frequency.agg({"Bearer Id": "count"})
#         Top10_sessions_frequency = sessions_frequency.sort_values(by='Bearer Id', ascending=False)
#         st.write(Top10_sessions_frequency.head(5))
#     elif a == 'duration of the session':
#         # duration of the session
#         st.write('top 5 duration of the session')
#         session_duration= data.groupby('MSISDN/Number')
#         session_duration = session_duration.agg({"Dur. (ms)": "sum"})
#         Top10_session_duration = session_duration.sort_values(by='Dur. (ms)', ascending=False)
#         st.write(session_duration.head(5))
#     elif a == 'sessions total traffic':
#         #the sessions total traffic (download and upload (bytes))
#         st.write('top 5 sessions total traffic (download and upload (bytes))')
#         total_traffic = data.groupby('MSISDN/Number')
#         total_traffic = total_traffic.agg({"Total": "sum"})
#         Top10_total_traffic = total_traffic.sort_values(by='Total', ascending=False)
#         st.write(Top10_total_traffic.head(5))
# def joined(data):
#     sessions_frequency = data.groupby('MSISDN/Number')
#     sessions_frequency = sessions_frequency.agg({"Bearer Id": "count"}) 
#     session_duration= data.groupby('MSISDN/Number')
#     session_duration = session_duration.agg({"Dur. (ms)": "sum"})
#     total_traffic = data.groupby('MSISDN/Number')
#     total_traffic = total_traffic.agg({"Total": "sum"}) 
#     return pd.DataFrame(sessions_frequency.join(session_duration, how='left')).join(total_traffic, how='left')

# def kmeans(data, k):
#     k = 3
#     joined_data = joined(data)
#     cols_to_standardize = ['Bearer Id',  'Dur. (ms)', 'Total']
#     data_to_standardize = joined_data[cols_to_standardize]

#     # Create the scaler.
#     scaler = StandardScaler().fit(data_to_standardize)

#     # Standardize the data
#     standardized_data = joined_data.copy()
#     standardized_columns = scaler.transform(data_to_standardize)
#     standardized_data[cols_to_standardize] = standardized_columns

#     st.write('Sample of data to use:')
#     st.write(standardized_data.sample(5), '\n')

#     model = KMeans(n_clusters = k).fit(standardized_data)

#     joined_data['cluster'] = model.predict(standardized_data)

#     st.write('Cluster summary:')
#     summary = joined_data.groupby(['cluster']).mean()
#     summary['count'] = joined_data['cluster'].value_counts()
#     summary = summary.sort_values(by='count', ascending=False)
#     st.write(summary)

# def most_app_used(data):
#     all_application = data[[ 'Social Media','Google', 'Email','Youtube','Netflix','Gaming']].sum()
#     all_ap = dict(all_application)
#     all_sum = dict(sorted(all_ap.items(), key=lambda item: item[1], reverse=True))
#     top3, i = {}, 0
#     for k, v in all_sum.items():
#         if i == 3: break
#         i+=1
#         top3[k] = v 
#     #Get the Keys and store them in a list
#     labels = list(top3.keys())
#     # Get the Values and store them in a list
#     values = list(top3.values())
#     #plt.pie(values, labels=labels, autopct='%1.2f%%')
#    # plt.show()

#     pie_fig = px.pie(values=values, names=labels)
#     st.plotly_chart(pie_fig)

# st.title("Upload and Download description")
# updl = st.selectbox('Choose type of web', ('Email','Social media','Google', 'Youtube', 'Netflix', 'Gaming'))
# total_data_description(data, updl)

# st.title('User Engagement analysis')
# analysis = st.selectbox('Choose customers per engagement metric', 
#         ('sessions frequency','duration of the session','sessions total traffic'))
# total_data_description(data, analysis)

# st.title('K-means (k=3) to classify customers in three groups of engagement.')
# kmeans(data, 3)

# st.title('Top 3 most used application')
# most_app_used(data)