# Project Analysis

This project includes a comprehensive analysis of soybean futures prices and weather data. It involves data visualization, time series decomposition, statistical testing, model building, and prediction.

## Case1 Analysis - Soybean Price and Weather Data Analysis

The `Case1 analysis.ipynb` notebook contains a detailed analysis of soybean prices and weather data from 2020 to 2022.

### Key Components:

1. **Data Loading and Preparation:**
   - Soybean futures data and weather data for the years 2020-2022 are loaded and concatenated.
   - The data is visualized to understand the trends over the years.

2. **Soybean Price Analysis:**
   - Seasonality in soybean prices is examined using time series decomposition.
   - The distribution of prices is visualized, and different seasonal prices are compared.

3. **Statistical Analysis:**
   - Returns of soybean prices are calculated and analyzed for normality.
   - Log returns are also examined, leading to the use of Geometric Brownian Motion for modeling.

4. **Modeling and Prediction:**
   - Geometric Brownian Motion model is used to predict future soybean prices.
   - A correlation analysis between weather indices and soybean prices is performed.
   - A combined model incorporating weather data is used for improved predictions.

5. **Futures and ETF Analysis:**
   - Futures data is analyzed and manually priced to correct discrepancies.
   - An ETF price series is constructed based on the manually priced futures data.

## GRU/LSTM Model - Deep Learning for Time Series Prediction

The `gru_lstm.ipynb` notebook explores deep learning models for time series prediction using GRU and LSTM neural networks.

### Key Components:

1. **Data Preprocessing:**
   - Data is prepared with time lags and one-hot encoding of days to be used as features for the neural networks.

2. **Modeling:**
   - LSTM and GRU models are built using PyTorch.
   - The models are trained and validated on the processed data.

3. **Evaluation:**
   - The models' performances are evaluated using the test dataset.
   - Predictions are formatted and inverse transformed to compare against actual values.

4. **Metrics:**
   - Performance metrics such as MAE, RMSE, and R^2 are calculated to assess the models' accuracy.

## SARIMA Model - Seasonal ARIMA for Time Series Analysis

The `sarimac.ipynb` notebook implements a Seasonal ARIMA model to analyze and predict time series data.

### Key Components:

1. **Data Visualization:**
   - ACF and PACF plots are generated to identify autocorrelation in the data.

2. **Model Building:**
   - The SARIMA model is configured with identified order and seasonal components.

3. **Prediction:**
   - The SARIMA model is used to make out-of-sample predictions.
   - Predictions are visualized alongside the original and seasonally adjusted data.

4. **Performance Evaluation:**
   - The Mean Squared Error (MSE) is computed to quantify the prediction accuracy.

---

Each notebook is designed to be self-contained, providing insights into different aspects of the data analysis process. For further details, refer to the respective notebooks and their in-depth analysis.
