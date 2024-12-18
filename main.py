import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Streamlit app title
st.title("Linear Regression App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df)
    
    # Select features and target variable
    st.sidebar.header("Model Configuration")
    features = st.sidebar.multiselect("Select Features", df.columns)
    target = st.sidebar.selectbox("Select Target Variable", df.columns)
    
    if features and target:
        # Splitting data
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Results
        st.subheader("Model Performance")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        
        # Coefficients
        st.subheader("Model Coefficients")
        coeff_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        })
        st.write(coeff_df)
