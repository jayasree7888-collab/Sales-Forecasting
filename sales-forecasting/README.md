# Sales-Forecasting-Dashboard-Streamlit-Machine-Learning-
This project is an interactive sales forecasting web application built using Python, XGBoost, and Streamlit. It predicts future sales trends based on historical data and provides clear, dynamic visualizations.
Users can upload their own sales CSV files, retrain the model directly in the browser with adjustable hyperparameters, and visualize predictions with confidence intervals.
The project is designed to help businesses make data-driven decisions by forecasting future sales performance efficiently.

🚀 Key Features

✅ Interactive Streamlit Dashboard – view, analyze, and forecast sales data in real-time.
✅ Machine Learning Model (XGBoost) – trained on historical time series sales data.
✅ Confidence Intervals (95%) – visualize prediction uncertainty with shaded intervals.
✅ Model Retraining – upload your own CSV file and retrain the model with custom hyperparameters.
✅ Download Forecasts – export predictions as a CSV file.
✅ Backtest RMSE Display – evaluate model accuracy on recent data.
✅ Clean Visualization – built with Matplotlib for a professional presentation.

⚙️ Tech Stack

Python 3.10+

Streamlit – UI framework for interactive dashboards

XGBoost – gradient boosting algorithm for time series forecasting

Scikit-learn – metrics and preprocessing

Matplotlib – data visualization

Pandas & NumPy – data handling and feature engineering

📦 Installation

Clone the repository:

git clone https://github.com/<your-username>/sales-forecast-dashboard.git
cd sales-forecast-dashboard


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


Then open your browser and go to 👉 http://localhost:8501

📊 Usage

Upload your historical sales data (date, sales) in CSV format.

Choose forecast horizon (1–12 months) using the sidebar slider.

Optionally, retrain the model using custom hyperparameters.

View forecast plots with confidence intervals.

Download your forecast results as a CSV file.

🧩 Example Input CSV
date,sales
2020-01-01,1540
2020-02-01,1600
2020-03-01,1625
2020-04-01,1580
2020-05-01,1655
...

📈 Example Output

Line chart showing historical vs predicted sales

Forecast for the next N months

RMSE metric for model accuracy

Downloadable CSV with predictions

🏗️ Project Structure
sales-forecasting/
├── app.py                     # Streamlit web dashboard
├── monthly_sales_forecast.py  # Model training and forecasting script
├── load_and_eda.py            # Data loading and EDA
├── processed_sales.csv        # Feature-engineered dataset
├── sales_forecast_model.pkl   # Trained XGBoost model
├── synthetic_monthly_sales.csv# Example dataset
└── requirements.txt           # Dependencies

📘 Future Enhancements

✅ Deploy to Streamlit Cloud / HuggingFace Spaces

✅ Auto hyperparameter tuning

✅ Email/Report generation for forecasts

✅ Integration with Google Sheets or SQL database

❤️ Author

Developed by [jaya sree]
If you like this project, please ⭐ the repository and share your feedback!

