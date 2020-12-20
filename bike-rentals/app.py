import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn import preprocessing
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import streamlit as st
import base64


DATA_LOC = 'data/bikerides_day.csv'

def load_data(folder):
    df = pd.read_csv(folder,error_bad_lines=False, encoding= 'unicode_escape', delimiter=',')
    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = ['ds', 'y', 'rain', 'temp']
    return df

def plot_trend(df):
    fig = go.Figure()
    #Create and style traces
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Rides',))
    return st.write(fig)

def box_cox_transform(df):
    df['y'], lam = boxcox(df['y'])
    return df, lam

def normalize(df):
    scaler = preprocessing.MinMaxScaler()
    bikerides_source = df.set_index('ds')
    bikerides_norm = scaler.fit_transform(bikerides_source)
    bikerides_norm = pd.DataFrame(bikerides_norm,columns=bikerides_source.columns, index=bikerides_source.index)
    bikerides_norm.reset_index(inplace=True)
    return bikerides_norm

def getPerformanceMetrics(m):
  return performance_metrics(getCrossValidationData(m))

def getCrossValidationData(m):
 return cross_validation(m, initial='730 days', period = '31 days', horizon = '365 days')

st.title('Forecasting Bikes Rides with FBProphet')
'''
This app uses Facebook's open-source Prophet library to generate forecast values of Bikes Riding Data.

Created by Bima Putra Pratama
'''

# Preparing Dataset

df = load_data(DATA_LOC)

bikerides,lam = box_cox_transform(df)
st.header('Dataset')

st.markdown('Select if you want to include weekend trend on the calculation')
if st.checkbox('Include Weekend'):
    weekend = True 
else:
    weekend = False
st.write(bikerides)



st.header('Ride volume over time')

if weekend == True:
    st.subheader('Includes Weekend')
    plot_trend(bikerides)

else:
    # Removing weekends
    bikerides.set_index('ds', inplace=True)
    bikerides = bikerides[bikerides.index.dayofweek < 5].reset_index()
    #Plot
    plot_trend(bikerides)

st.header('Visualize Rides with Temperature and Rain')

'''
Let see how temperature and rain affects number of ride. This graph shows normalized value of each variables
'''

bikerides_norm = normalize(bikerides)
#Plotting
fig = go.Figure()
#Create and style traces
fig.add_trace(go.Scatter(x=bikerides_norm['ds'], y=bikerides_norm['y'], name='Rides',))
fig.add_trace(go.Scatter(x=bikerides_norm['ds'], y=bikerides_norm['rain'], name='Rain',))
fig.add_trace(go.Scatter(x=bikerides_norm['ds'], y=bikerides_norm['temp'], name='Temp',))
st.write(fig)

## Model Training

st.header('Model Training')
st.subheader('Model Configuration')

# read param

holiday_con = False
rain_con = False
temp_con = False

if st.checkbox('Add Holiday'):
    holiday_con = True 

if st.checkbox('Consider Rain'):
    rain_con = True 

if st.checkbox('Consider Temp'):
    temp_con = True 

# Forecasting

st.subheader('Model Results and Evaluation')

m = Prophet(
    interval_width = 0.9,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

if holiday_con == True:
    ascensionday = pd.DataFrame({
        'holiday': 'AscensionDay',
        'ds': pd.to_datetime(['2019-05-30']),
        'lower_window': 0,
        'upper_window': 1,
        })

    christmas = pd.DataFrame({
        'holiday': 'Christmas',
        'ds': pd.to_datetime(['2017-12-24','2018-12-24','2019-12-24','2020-12-24']),
        'lower_window': -1,
        'upper_window': 7,
        })

    holidays = pd.concat((ascensionday, christmas))

    m = Prophet(
        holidays=holidays,
        interval_width = 0.9,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False)
    m.add_country_holidays(country_name='NO')  

if rain_con==True:
    m.add_regressor('rain')

if temp_con==True:
    m.add_regressor('temp')

# Fit the data. Remember that prophet expect "ds" and "y" as names for the columns.
m.fit(bikerides)

# We must create a data frame holding dates for our forecast. The periods # parameter counts days as long as the frequency is 'D' for the day. Let's # do a 180 day forecast, approximately half a year.
future = m.make_future_dataframe(periods=0, freq='D')

future = future.merge(bikerides, on='ds')

forecast = m.predict(future)

fig = go.Figure()

# Create and style traces
fig.add_trace(go.Scatter(x=bikerides['ds'], y=bikerides['y'], name='Actual',))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted',))

if rain_con==True:
    fig.add_trace(go.Scatter(x=bikerides['ds'], y=bikerides['rain'], name='Rain',))

if temp_con==True:
    fig.add_trace(go.Scatter(x=bikerides['ds'], y=bikerides['temp'], name='Temp',))

st.write(fig)

st.write(m.plot(forecast))
st.write(m.plot_components(forecast))

st.write(getPerformanceMetrics(m).mean())

st.header('Forecasting Rides')
period = st.slider('Forecast Periods in days',1,365,)
st.write("Forecast Periods ", period, " days")

# We must create a data frame holding dates for our forecast. The periods # parameter counts days as long as the frequency is 'D' for the day. Let's # do a 180 day forecast, approximately half a year.
future = m.make_future_dataframe(periods=period, freq='D')

future = future.merge(bikerides, on='ds', how = 'left')

# Add default values for now. Can be expanded by forecastting each components
future['rain']=future['rain'].fillna(value=0)
future['temp']=future['temp'].fillna(value=bikerides['temp'].mean())

# Create the forecast object which will hold all of the resulting data from the forecast.
forecast = m.predict(future)
# st.write(forecast)

# Transform back to reality from Box Cox
forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))
bikerides['y'] = inv_boxcox(bikerides['y'], lam)

fig = go.Figure()
#Create and style traces
fig.add_trace(go.Scatter(x=bikerides['ds'], y=bikerides['y'], name='Actual',))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted',))
st.write(fig)

st.write(forecast[['ds','yhat','yhat_lower','yhat_upper']])

"""
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

max_date = bikerides['ds'].max()
fcst_filtered = forecast[forecast['ds'] > max_date] 
fcst_filtered = fcst_filtered[['ds','yhat','yhat_lower','yhat_upper']]

csv_exp = fcst_filtered.to_csv(index=False)
# When no file name is given, pandas returns the CSV as a string, nice.
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)