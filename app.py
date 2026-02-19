import streamlit as st
import os

st.title("Debug Mode")

st.write("Current working directory:")
st.write(os.getcwd())

st.write("Files in current directory:")
st.write(os.listdir())
