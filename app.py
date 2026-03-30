import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("traffic_detections.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

st.title("🚗 Traffic Vehicle Analytics Dashboard")
st.markdown("Real-time traffic analysis powered by **YOLOv8** object detection pipeline")
st.divider()

# Sidebar filters
st.sidebar.header("Filters")
days = st.sidebar.multiselect(
    "Select days",
    options=df["day_of_week"].unique().tolist(),
    default=df["day_of_week"].unique().tolist()
)
hour_range = st.sidebar.slider("Hour range", 0, 23, (0, 23))
vehicle_types = st.sidebar.multiselect(
    "Vehicle types",
    options=["car", "truck", "bus", "motorcycle"],
    default=["car", "truck", "bus", "motorcycle"]
)

# Filter data
filtered = df[
    (df["day_of_week"].isin(days)) &
    (df["hour"] >= hour_range[0]) &
    (df["hour"] <= hour_range[1])
]

# KPI metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total vehicles", f"{filtered['total_vehicles'].sum():,}")
col2.metric("Avg per 15 min", f"{filtered['total_vehicles'].mean():.1f}")
col3.metric("Peak hour", f"{filtered.groupby('hour')['total_vehicles'].mean().idxmax()}:00")
col4.metric("Records", f"{len(filtered):,}")

st.divider()

# Row 1 — Hourly + Pie
col1, col2 = st.columns([2, 1])

with col1:
    hourly = filtered.groupby("hour")["total_vehicles"].mean().reset_index()
    fig = px.bar(
        hourly, x="hour", y="total_vehicles",
        title="Average Traffic Volume by Hour",
        labels={"hour": "Hour of day", "total_vehicles": "Avg vehicles"},
        color_discrete_sequence=["#1D9E75"]
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    type_totals = filtered[vehicle_types].sum().reset_index()
    type_totals.columns = ["type", "count"]
    fig2 = px.pie(
        type_totals, names="type", values="count",
        title="Vehicle Type Distribution",
        color_discrete_sequence=["#1D9E75", "#185FA5", "#E74C3C", "#EF9F27"]
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

# Row 2 — Time series + Daily
col3, col4 = st.columns([2, 1])

with col3:
    fig3 = px.line(
        filtered, x="timestamp", y="total_vehicles",
        title="Traffic Flow Over Time",
        labels={"timestamp": "Time", "total_vehicles": "Vehicles"},
        color_discrete_sequence=["#185FA5"]
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    daily = filtered.groupby(["date", "is_weekend"])["total_vehicles"].sum().reset_index()
    daily["type"] = daily["is_weekend"].map({True: "Weekend", False: "Weekday"})
    fig4 = px.bar(
        daily, x="date", y="total_vehicles", color="type",
        title="Daily Traffic Volume",
        color_discrete_map={"Weekday": "#1D9E75", "Weekend": "#B4B2A9"}
    )
    fig4.update_layout(height=350, xaxis_tickangle=45)
    st.plotly_chart(fig4, use_container_width=True)

# Raw data table
st.divider()
st.subheader("Raw detection data")
st.dataframe(
    filtered[["timestamp", "day_of_week", "total_vehicles",
              "car", "truck", "bus", "motorcycle"]]
    .sort_values("timestamp", ascending=False)
    .head(100),
    use_container_width=True
)