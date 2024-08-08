import streamlit as st
from forecasting import show_forecasting_page
from chatbot import show_chatbot_page

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['📈Forecasting', '💬Chatbot'])

# Show the selected page
if page == '📈Forecasting':
    show_forecasting_page()
elif page == '💬Chatbot':
    show_chatbot_page()
