import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import pickle

def show_forecasting_page():
    # Header section with clean styling
    st.title('🛢️ Oil Production Forecasting')
    
    # Information section in a clean card-like container
    with st.container():
        st.markdown("""
        <div style='background-color: rgba(35, 45, 55, 0.8); padding: 20px; border-radius: 10px;'>
        <h3>About the Forecasting Model</h3>
        <p>This forecasting tool uses Facebook Prophet, an advanced time series forecasting model that excels in:</p>
        <ul>
        <li>Handling missing values and outliers in production data</li>
        <li>Capturing seasonal patterns and trend changes</li>
        <li>Providing robust predictions for petroleum production</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Load the pickled model
    with open(r'prophet_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Function to make predictions
    def make_predictions(periods):
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast

    # Create two columns for inputs and key metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User input with better formatting
        st.markdown("### Forecast Settings")
        periods = st.slider(
            'Forecast Horizon (Days)',
            min_value=1,
            max_value=365,
            value=30,
            help="Select the number of days you want to forecast into the future"
        )

    # Make predictions
    forecast = make_predictions(periods)
    
    with col2:
        # Display key metrics
        st.markdown("### Key Metrics")
        last_value = forecast['yhat'].iloc[-periods-1]
        future_value = forecast['yhat'].iloc[-1]
        change = ((future_value - last_value) / last_value) * 100
        
        st.metric(
            label="Predicted Change",
            value=f"{change:.1f}%",
            delta=f"{future_value-last_value:.1f} units"
        )

    # Plot results with enhanced styling
    st.markdown("### Production Forecast Visualization")
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=forecast['ds'][:len(forecast)-periods],
        y=forecast['yhat'][:len(forecast)-periods],
        mode='lines',
        name='Historical Forecast',
        line=dict(color='#2E86C1', width=2)
    ))

    # Plot future predictions
    fig.add_trace(go.Scatter(
        x=forecast['ds'][len(forecast)-periods:],
        y=forecast['yhat'][len(forecast)-periods:],
        mode='lines',
        name='Future Forecast',
        line=dict(color='#28B463', width=2)
    ))

    # Update layout with better styling
    fig.update_layout(
        title={
            'text': 'Oil Production Forecast',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='Production',
        template='plotly_dark',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.1)'
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast table in an expander with better formatting
    with st.expander("📊 Detailed Forecast Data"):
        st.markdown("### Forecasted Production Values")
        
        # Extract and rename forecasted values
        forecast_table = forecast[['ds', 'yhat']].tail(periods)
        forecast_table = forecast_table.rename(columns={
            'ds': 'Date',
            'yhat': 'Predicted Production'
        })
        
        # Format the values
        forecast_table['Predicted Production'] = forecast_table['Predicted Production'].round(2)
        forecast_table['Date'] = pd.to_datetime(forecast_table['Date']).dt.strftime('%Y-%m-%d')
        
        # Add styling to the table
        st.dataframe(
            forecast_table.style.background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        # Add download button for the forecast data
        st.download_button(
            label="Download Forecast Data",
            data=forecast_table.to_csv(index=False),
            file_name="oil_production_forecast.csv",
            mime="text/csv"
        )

    # Additional information footer
    st.markdown("""
    <div style='background-color: rgba(35, 45, 55, 0.8); padding: 15px; border-radius: 10px; margin-top: 20px;'>
    <h4>📝 Notes</h4>
    <p>The forecast is based on historical production data and uses Prophet's advanced capabilities to provide accurate predictions.
    Consider the confidence intervals when making decisions based on these forecasts.</p>
    </div>
    """, unsafe_allow_html=True)
    

# ====================================================================

# ===========================================================================
# The below code is to run the page alone
# ===========================================================================
       
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