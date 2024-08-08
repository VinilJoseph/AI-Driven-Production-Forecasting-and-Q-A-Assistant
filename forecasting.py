# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# from prophet import Prophet
# import pickle

# # Load the pickled model
# with open(r'C:\Users\DELL\Production_forecasting_strmlt_prjct\prophet_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Function to make predictions
# def make_predictions(periods):
#     future = model.make_future_dataframe(periods=periods)
#     forecast = model.predict(future)
#     return forecast

# # Streamlit UI
# st.title('Oil Production Forecasting with Prophet')

# # User input for forecasting period
# periods = st.number_input('Enter number of days for forecasting:', min_value=1, value=30)

# # Make predictions
# forecast = make_predictions(periods)

# # Plot results
# fig = go.Figure()

# # Plot historical data
# fig.add_trace(go.Scatter(x=forecast['ds'][:len(forecast)-periods], y=forecast['yhat'][:len(forecast)-periods], mode='lines', name='Historical Forecast'))

# # Plot future predictions
# fig.add_trace(go.Scatter(x=forecast['ds'][len(forecast)-periods:], y=forecast['yhat'][len(forecast)-periods:], mode='lines', name='Future Forecast'))

# # Update layout
# fig.update_layout(title='Oil Production Forecasting with Prophet',
#                   xaxis_title='Date',
#                   yaxis_title='Production',
#                   template='plotly_dark',
#                   legend=dict(x=0, y=1))

# st.plotly_chart(fig)


# # # Display forecasted values in a table
# # st.subheader('Forecasted Values')

# # # Extract and rename forecasted values
# # forecast_table = forecast[['ds', 'yhat']].tail(periods)
# # forecast_table = forecast_table.rename(columns={'ds': 'Date', 'yhat': 'Production'})

# # # Reset index to start from 1
# # forecast_table.reset_index(drop=True, inplace=True)
# # forecast_table.index += 1

# # # Display the table
# # st.table(forecast_table)


# # Display forecasted values in a collapsible expander
# st.subheader('Forecasted Values')

# with st.expander("Show Forecast Table"):
#     # Extract and rename forecasted values
#     forecast_table = forecast[['ds', 'yhat']].tail(periods)
#     forecast_table = forecast_table.rename(columns={'ds': 'Date', 'yhat': 'Production'})

#     # Reset index to start from 1
#     forecast_table.reset_index(drop=True, inplace=True)
#     forecast_table.index += 1

#     # Display the table
#     st.table(forecast_table)







import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import pickle

def show_forecasting_page():
    # Load the pickled model
    with open(r'C:\Users\DELL\Production_forecasting_strmlt_prjct\prophet_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Function to make predictions
    def make_predictions(periods):
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast

    # Streamlit UI
    st.title('Oil Production Forecasting with Prophet')

    # User input for forecasting period
    periods = st.number_input('Enter number of days for forecasting:', min_value=1, value=30)

    # Make predictions
    forecast = make_predictions(periods)

    # Plot results
    fig = go.Figure()


    # Plot historical data
    fig.add_trace(go.Scatter(x=forecast['ds'][:len(forecast)-periods], y=forecast['yhat'][:len(forecast)-periods], mode='lines', name='Historical Forecast'))

    # Plot future predictions
    fig.add_trace(go.Scatter(x=forecast['ds'][len(forecast)-periods:], y=forecast['yhat'][len(forecast)-periods:], mode='lines', name='Future Forecast'))

    # Update layout
    fig.update_layout(title='Oil Production Forecasting with Prophet',
                      xaxis_title='Date',
                      yaxis_title='Production',
                      template='plotly_dark',
                      legend=dict(x=0, y=1))

    st.plotly_chart(fig)
    

    # Display forecasted values in a collapsible expander
    st.subheader('Forecasted Values')

    with st.expander("Show Forecast Table"):
        # Extract and rename forecasted values
        forecast_table = forecast[['ds', 'yhat']].tail(periods)
        forecast_table = forecast_table.rename(columns={'ds': 'Date', 'yhat': 'Production'})

        # Reset index to start from 1
        forecast_table.reset_index(drop=True, inplace=True)
        forecast_table.index += 1

        # Display the table
        st.table(forecast_table)