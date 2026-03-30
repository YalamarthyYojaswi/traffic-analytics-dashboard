# Traffic Vehicle Analytics Dashboard

An end-to-end pipeline combining YOLOv8 vehicle detection with an 
interactive analytics dashboard built with Streamlit and Plotly.

## Live Dashboard

Try it here: https://traffic-analytics-dashboard-yoji.streamlit.app/

## What this project does

- Runs YOLOv8 on traffic camera images to detect and classify vehicles
- Processes detection results into time-series traffic data
- Visualizes patterns in an interactive Streamlit dashboard with filters

## Dashboard Features

- KPI metrics (total vehicles, avg per interval, peak hour)
- Hourly traffic volume bar chart
- Vehicle type distribution pie chart
- Traffic flow time series
- Weekday vs weekend comparison
- Filterable by day, hour range, and vehicle type
- Raw detection data table

## Tech Stack

- Python, YOLOv8 (Ultralytics)
- Streamlit, Plotly, pandas
- Kaggle (T4 GPU for detection)
- Streamlit Cloud (deployment)
