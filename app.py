
import streamlit as st
from forecasting import show_forecasting_page
from chatbot import show_chatbot_page

# Set page config at the very beginning
st.set_page_config(
    page_title="Oil Production Forecasting",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
st.sidebar.title('Navigation')
st.sidebar.markdown("""
---
Choose a service:
""")

page = st.sidebar.radio('', [
    '📈 Forecasting',
    '💬 Chatbot'
])

# Brief instructions under the radio button
if page == '📈 Forecasting':
    st.sidebar.info("""
    **Forecasting Service**
      - Enter time range to forecast
      - Get predictions using Prophet model
      - Visualize future trends
      - Analyse table for daily data
    """)
elif page == '💬 Chatbot':
    st.sidebar.info("""
    **Chatbot Assistant**
    - Ask questions about the field and equipments
    - Get instant answer
    - Technical support
    """)

# Show the selected page
if page == '📈 Forecasting':
    show_forecasting_page()
elif page == '💬 Chatbot':
    show_chatbot_page()
# ===================================================

# import streamlit as st
# from forecasting import show_forecasting_page
# from chatbot import show_chatbot_page

# # Sidebar for navigation
# st.sidebar.title('Navigation')
# page = st.sidebar.radio('Go to', ['📈Forecasting', '💬Chatbot'])

# # Show the selected page
# if page == '📈Forecasting':
#     show_forecasting_page()
# elif page == '💬Chatbot':
#     show_chatbot_page()




# ===================================================
