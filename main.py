import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import bisect
from datetime import datetime

st.set_page_config('Dashboard', layout="wide")
st.title('Pharmaceutical Sales prediction across multiple stores') 
store_data = pd.read_csv('./data/store.csv', na_values=['?', None, 'undefined'])
train_data = pd.read_csv('./data/train.csv', na_values=['?', None, 'undefined'])
test_data = pd.read_csv('./data/test.csv', na_values=['?', None, 'undefined'])
#submission_data = pd.read_csv('./data/sample_submission.csv', na_values=['?', None, 'undefined'])
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
