"""
Issue: python-magic
> https://github.com/Yelp/elastalert/issues/1927
"""

# Local packages
from pkg.signal import get_file_content, get_duration, \
    apply_smoothing, apply_tkeo_operator

# External packages
import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Myodev - Signal analysis",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Settings")
file_input = st.sidebar.file_uploader("Upload the signal (.txt)")
st.sidebar.header("Signal parameters")
threshold_level = st.sidebar.slider("Threshold level", value=10, min_value=0, max_value=100, step=10, format="%d%%")
smooth_level = st.sidebar.slider("Smooth level", value=20, min_value=0, max_value=100, step=10, format="%d%%")

st.sidebar.header("What to display")
plot_original = st.sidebar.checkbox("Original signal", value=True)
plot_tkeo = st.sidebar.checkbox("TKEO signal", value=True)
plot_smoothed = st.sidebar.checkbox("Smoothed signal", value=True)

st.title('Myodev - Signal analysis')
col1, col2 = st.columns(2)
col1.metric("Threshold level", f"{threshold_level}%", None)
col2.metric("Smooth level", f"{smooth_level}%", None)

if file_input is not None:
    data = get_file_content(file_input)
    chart_data = pd.DataFrame(None)
    signal = data['A1']
    tkeo_signal = apply_tkeo_operator(signal)
    smoothed_signal = apply_smoothing(tkeo_signal, 1000, smooth_level_perc=smooth_level)
    duration = get_duration(signal, threshold_level=threshold_level, smooth_level=smooth_level)

    if plot_original:
        chart_data['Original signal'] = signal

    if plot_tkeo:
        chart_data['TKEO signal'] = tkeo_signal

    if plot_smoothed:
        chart_data['Smoothed signal'] = smoothed_signal

    st.line_chart(chart_data)

    cols = st.columns(4)
    cols[0].metric("No. contractions", duration['no_contraction'], None)
    cols[1].metric("Average time", round(duration['avg'], 2), None)
    cols[2].metric("Standard deviation time", round(duration['std'], 2), None)

else:
    st.write("Waiting data...")

st.download_button('Download data', "fff")
st.button('Send data')