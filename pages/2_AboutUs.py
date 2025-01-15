import streamlit as st
import sklearn

# About Us page
st.markdown(
    """
    <div style='display: flex; text-align: center; flex-direction: column'>
        <h1>About Us</h1>
        <br/><br/>
        <p style='font-size: 18px; max-width: 600px; margin: auto;'>
            PayUp is our web application designed to predict user salaries based on their entered details. 
            We utilize the StackOverflow 2023 developer survey dataset to provide accurate and reliable predictions.
        </p>
        <p style='font-size: 18px; max-width: 600px; margin: auto;'>
            Thank you for using PayUp! We hope our application helps you in understanding your potential salary.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
