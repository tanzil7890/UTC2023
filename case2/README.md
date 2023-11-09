# Options Trading Analysis and Strategy

This project includes an analysis of options trading data and the implementation of a trading strategy using Python.

## Case 2 Analysis - `case2.ipynb`

The `case2.ipynb` notebook contains an in-depth analysis of options pricing and trading strategies based on the Black-Scholes model and other financial concepts.

### Key Components:

1. **Data Loading:**
   - Loading price paths and training data from CSV files.

2. **Data Visualization:**
   - Plotting asset prices and options prices to visualize the trends and patterns.
  
3. **Options Pricing:**
   - Implementing functions to calculate the Greeks and the prices of call and put options using Black-Scholes formulas.

4. **Binomial Tree Pricing Model:**
   - Defining classes for European and Binomial LR options pricing.

5. **Trading Strategy Implementation:**
   - Implementing a function to add trading strategies based on the options pricing and Greeks.

6. **Visualization:**
   - Plotting histograms to understand the distribution of asset prices and using matplotlib for various other plots.

## GRU/LSTM Model - `gru_lstm_options.ipynb`

This notebook explores the use of GRU and LSTM neural networks to predict options pricing.

### Key Components:

1. **Data Preparation:**
   - Generating time-lagged features for use in neural network training.

2. **Model Training:**
   - Training GRU and LSTM models using PyTorch for time series prediction.

3. **Model Evaluation:**
   - Evaluating model performance and plotting training/validation losses.

4. **Prediction Formatting:**
   - Formatting the predictions and comparing them against actual values.

5. **Metrics Calculation:**
   - Calculating Mean Absolute Error, Root Mean Squared Error, and R^2 Score for model evaluation.

## Understanding Training Data - `understanding_train_data.ipynb`

This notebook includes exploratory data analysis of training data for options trading.

### Key Components:

1. **Statistical Analysis:**
   - Running Augmented Dickey-Fuller tests to check for stationarity.

2. **Correlogram:**
   - Plotting ACF and PACF to understand autocorrelations in the time series data.

3. **Implied Volatility:**
   - Analyzing and visualizing the implied volatility surface for different strikes and maturities.

4. **Greeks Analysis:**
   - Calculating and visualizing the Greeks for a range of strike prices.

## Trading Bot Implementation - `trader_bot.py`

A Python script that contains the implementation of an options trading bot.

### Key Components:

1. **Bot Initialization:**
   - Setting up initial positions and parameters for the trading bot.

2. **Market Updates:**
   - Handling updates from the exchange and adjusting the bot's strategy accordingly.

3. **Trading Strategy:**
   - Defining and adjusting the bot's trading strategy based on market conditions and the bot's current positions.

4. **Execution:**
   - Sending orders to the exchange and updating the bot's strategy based on the responses.

---

Please refer to the respective notebooks and script for more details and the full context of the code.
