# Bitcoin_ClosingPrice_Prediction
This project predicts Bitcoin's closing price using machine learning models such as XGBoost and GRU. The dataset consists of historical Bitcoin prices from 2020 to 2025, with preprocessing and feature engineering applied to improve model accuracy.

📌 Features
Data Preprocessing: Handles missing values, removes duplicates, and scales numerical features.

Feature Engineering: Extracts day, month, and year from the date column.

Machine Learning Models:

XGBoost: Trained as a baseline model.

GRU (Gated Recurrent Unit): Used for sequence-based predictions.

Performance Evaluation: Computes MAE, MSE, RMSE, and R² score.

Visualization: Plots actual vs. predicted Bitcoin prices.

📂 Dataset
The dataset is a CSV file containing Bitcoin historical data.

It includes columns like Open, High, Low, Close, Volume, and Date.

The Close price is the target variable.

🛠️ Installation
To run this project, install the necessary dependencies:


pip install numpy pandas scikit-learn xgboost tensorflow matplotlib seaborn

🏆 Future Improvements
Integrate LSTM and BiLSTM for better accuracy.

Allow predictions for a single day without requiring 50 days of prior data.

Implement an API for real-time predictions.

Use external market indicators (e.g., trading volume, economic factors).

🤝 Contributing
Feel free to fork this repo, open an issue, or submit a pull request to improve the project.
