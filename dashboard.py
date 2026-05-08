import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#Dashboard Title
st.title("Retail Sales Forecasting Dashboard")

#Description
st.write("Predict Walmart Weekly Sales using Machine Learning")

#Load Dataset
df = pd.read_csv("Walmart_Sales.csv")

#Changing the format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extract useful temporal features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

#SIDEBAR INPUTS
st.sidebar.header("Input Features")

store = st.sidebar.number_input(
    "Store",
    min_value=1
)

holiday = st.sidebar.selectbox(
    "Holiday Flag",
    [0,1]
)

temperature = st.sidebar.slider(
    "Temperature",
    float(df['Temperature'].min()),
    float(df['Temperature'].max())
)

fuel_price = st.sidebar.slider(
    "Fuel Price",
    float(df['Fuel_Price'].min()),
    float(df['Fuel_Price'].max())
)

cpi = st.sidebar.slider(
    "CPI",
    float(df['CPI'].min()),
    float(df['CPI'].max())
)

unemployment = st.sidebar.slider(
    "Unemployment",
    float(df['Unemployment'].min()),
    float(df['Unemployment'].max())
)

year = st.sidebar.selectbox(
    "Year",
    sorted(df['Year'].unique())
)

month = st.sidebar.slider(
    "Month",
    1,
    12
)

week = st.sidebar.slider(
    "Week",
    1,
    52
)

#Load Trained Model
model = pickle.load(open("model.pkl", "rb"))

#Input Data
input_data = pd.DataFrame({
    'Store':[store],
    'Holiday_Flag':[holiday],
    'Temperature':[temperature],
    'Fuel_Price':[fuel_price],
    'CPI':[cpi],
    'Unemployment':[unemployment],
    'Year':[year],
    'Month':[month],
    'Week':[week]
})

#Prediction Button
if st.button("Predict Sales"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Weekly Sales: ${prediction[0]:,.2f}"
    )

#Monthly Sales Trend
st.subheader("Monthly Sales Trend")

monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker='o'
)

ax.set_xlabel("Month")
ax.set_ylabel("Average Weekly Sales")
ax.set_title("Average Monthly Sales")

st.pyplot(fig)

#Holiday vs Non-Holiday Sales
st.subheader("Holiday vs Non-Holiday Sales")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.boxplot(
    x='Holiday_Flag',
    y='Weekly_Sales',
    data=df,
    ax=ax2
)

ax2.set_title("Holiday vs Non-Holiday Sales")

st.pyplot(fig2)

#Feature Correlation Heatmap
st.subheader("Feature Correlation")

fig3, ax3 = plt.subplots(figsize=(10,6))

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax3
)

st.pyplot(fig3)
