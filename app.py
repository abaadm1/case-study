import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from dateutil.parser import parse
# Load the trained KMeans model from the pickle file
with open('rf_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Function to extract features from time
def extract_time_features(time_str):
    time = pd.to_datetime(time_str)
    day = time.day
    month = time.month
    year = time.year
    dayofweek = time.dayofweek
    return day, month, year, dayofweek

# Function to prepare the new observation
def prepare_observation(location, time, amount, acquired_by, acquisition_time):
    transaction_day, transaction_month, transaction_year, transaction_dayofweek = extract_time_features(time)
    acquisition_day, acquisition_month, acquisition_year, acquisition_dayofweek = extract_time_features(acquisition_time)
    observation = pd.DataFrame({
        'value': [amount],
        'transaction_day': [transaction_day],
        'transaction_month': [transaction_month],
        'transaction_year': [transaction_year],
        'transaction_dayofweek': [transaction_dayofweek],
        'acquired_day': [acquisition_day],
        'acquired_month': [acquisition_month],
        'acquired_year': [acquisition_year],
        'acquired_dayofweek': [acquisition_dayofweek],
        'acquired_by_encoded': [acquired_by],
        'parent_region_id_encoded': [location]
    })
    new_observation_scaled = scaler.transform(observation)
    return new_observation_scaled


# Create a Streamlit app
def main():
    st.title("Customer Value Prediction")
    # Sidebar with input fields
    st.sidebar.title("Enter Customer Details")
    parent_region_id = st.sidebar.selectbox("Location", ["07B", "300", "1EF"])
    if parent_region_id == "07B":
        parent_region_id_encoded = 0
    elif parent_region_id == "1EF":
        parent_region_id_encoded = 1
    else:
        parent_region_id_encoded =2
    time = st.sidebar.text_input("Transaction Time (YYYY-MM-DD HH:MM:SS)")
    amount = st.sidebar.number_input("Transaction Amount", min_value=0, step=1)
    acquired_by = st.sidebar.selectbox("Acquired By", ["OFFLINE", "ONLINE"])
    acquired_by_encoded = 0 if acquired_by == "OFFLINE" else 1
    acquisition_time = st.sidebar.text_input("Acquisition Time (YYYY-MM-DD HH:MM:SS)")


    # Add a button for prediction
    predict_button = st.sidebar.button("Predict")

    if predict_button:
        # Prepare the new observation
        try:
            transaction_time = parse(time)
            acquisition_time = parse(acquisition_time)
        except Exception as e:
            st.error("Error occurred while parsing time: " + str(e))
            return

        new_observation = prepare_observation(parent_region_id_encoded, transaction_time, amount, acquired_by_encoded, acquisition_time)

        # Predict the cluster for the new observation
        with st.spinner("Predicting..."):
            predicted_cluster = kmeans_model.predict(new_observation)[0]
            if predicted_cluster ==0:
                cluster_name = "Low Value Customer"
            else:
                predicted_cluster = 1
                cluster_name = "High Value Customer"

        st.success(f"The Predicted Cluster is: {predicted_cluster} \nThis Customer is a {cluster_name} ")

# Run the Streamlit app
if __name__ == "__main__":
    main()