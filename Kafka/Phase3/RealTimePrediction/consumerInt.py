from kafka import KafkaConsumer
import json
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta

# File paths
output_file = "streamed_kafka_data.csv"
predictions_file = "hourly_predictions.csv"
model_path = "D:/Grace/CMU/Courses/Spring2025/OpAI/Assignment/1/integratedPipeline/xgb_model.pkl"

# Load trained model and feature list from pickle file
with open(model_path, "rb") as f:
    model_metadata = pickle.load(f)
model = model_metadata["model"]
training_features = model_metadata["features"]

# Kafka Consumer Initialization
consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

def clean_data(record):
    """
    Cleans a single record by replacing missing values (-200 or 'Unknown').
    
    Args:
        record (dict): A dictionary representing a single record.
    
    Returns:
        dict: Cleaned record.
    """
    try:
        df = pd.DataFrame([record])
        #df.replace([-200, '-200', 'Unknown'], pd.NA, inplace=True)
        #df.ffill(inplace=True)
        #df.bfill(inplace=True)
        #for col in df.columns:
            #if pd.api.types.is_numeric_dtype(df[col]):
                #df[col].fillna(df[col].mean(), inplace=True)
        return df.to_dict(orient='records')[0]
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None

def save_to_csv(record):
    """
    Saves a single cleaned record to a CSV file with proper column names.
    
    Args:
        record (dict): A dictionary representing a single cleaned record.
    """
    column_names = [
        'CO(GT)', 'PT08.S1(CO)', #'NMHC(GT)', 
        'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH', 'Datetime'
    ]
    
    try:
        # Convert the record into a DataFrame for saving
        df = pd.DataFrame([record])
        
        # Write headers only if the file does not exist or is empty
        write_header = not os.path.exists(output_file) or os.stat(output_file).st_size == 0
        
        # Save to CSV with proper headers
        df.to_csv(output_file, mode='a', header=write_header, index=False, columns=column_names)
    
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def preprocess_for_prediction(df):
    """
    Generate features for prediction based on historical hourly data.
    
    Args:
        df (DataFrame): Historical hourly data.
    
    Returns:
        DataFrame: Processed data with features for prediction.
    """
    for lag in range(1, 4):
        df[f'CO_lag_{lag}'] = df['CO(GT)'].shift(lag)
    
    rolling_window = 3
    df['CO_roll_mean'] = df['CO(GT)'].rolling(window=rolling_window).mean()
    df['CO_roll_std'] = df['CO(GT)'].rolling(window=rolling_window).std()
    
    processed_df = df.dropna()
    
    processed_df, _ = processed_df.align(pd.DataFrame(columns=training_features), axis=1, fill_value=0)
    
    return processed_df

def predict_next_hour(processed_df):
    """
    Predict the next hour's pollutant concentration using the trained model.
    
    Args:
        processed_df (DataFrame): Processed data with features for prediction.
    
    Returns:
        tuple: Predicted datetime and pollutant concentration.
    """
    last_datetime = pd.to_datetime(processed_df['Datetime'].iloc[-1])
    next_datetime = last_datetime + timedelta(hours=1)

    prediction_input = processed_df.iloc[-1][training_features].values.reshape(1, -1)
    
    predicted_value = model.predict(prediction_input)[0]
    
    return next_datetime, predicted_value

def save_prediction(datetime, value):
    """
    Save the hourly prediction to a CSV file.
    
    Args:
        datetime (datetime): Predicted datetime.
        value (float): Predicted pollutant concentration.
    """
    pd.DataFrame([[datetime, value]], columns=['Datetime', 'Predicted_CO']).to_csv(
        predictions_file,
        mode='a',
        header=not os.path.exists(predictions_file),
        index=False
    )

def consume_messages():
    """
    Consume messages from Kafka and predict next hour's pollutant concentration.
    """
    print("Consumer started...")
    
    for message in consumer:
        raw_record = message.value
        
        cleaned_record = clean_data(raw_record)
        
        if cleaned_record is None:
            print("Skipping invalid record.")
            continue
        
        save_to_csv(cleaned_record)
        
        try:
            full_df = pd.read_csv(output_file)
            full_df['Datetime'] = pd.to_datetime(full_df['Datetime'])
            
            processed_df = preprocess_for_prediction(full_df)

            if not processed_df.empty:
                next_datetime, predicted_value = predict_next_hour(processed_df)
                save_prediction(next_datetime, predicted_value)
                print(f"Predicted CO for {next_datetime}: {predicted_value:.2f}")
        
        except Exception as e:
            print(f"Prediction pipeline error: {e}")

if __name__ == '__main__':
    for f in [output_file, predictions_file]:
        if not os.path.exists(f):
            open(f, 'w').close()
    
    consume_messages()
