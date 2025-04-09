from kafka import KafkaConsumer
import json
import pandas as pd
import os

# Initialize the Kafka consumer
consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',  # Start reading from the beginning of the topic if no offset is stored
    enable_auto_commit=True,  # Automatically commit the message offset after it's read
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))  # Deserialize JSON messages
)

# Output file name
output_file = "streamed_kafka_data.csv"

# Function to clean data with missing values
def clean_data(record):
    """
    Cleans a single record by replacing missing values (-200 or 'Unknown').
    
    Args:
        record (dict): A dictionary representing a single record.
    
    Returns:
        dict: Cleaned record.
    """
    try:
        # Convert the record to a pandas DataFrame for easier manipulation
        df = pd.DataFrame([record])
        
        # Replace -200 with NaN for numerical columns
        df.replace(-200, pd.NA, inplace=True)
        df.replace('-200', pd.NA, inplace=True)
        
        # Replace 'Unknown' with NaN for datetime or string columns
        df.replace('Unknown', pd.NA, inplace=True)
        
        # Forward fill missing values if possible (column-wise)
        df = df.ffill(axis=0)
        
        # Backfill missing values if forward fill is not possible (column-wise)
        df = df.bfill(axis=0)
        
        # Replace any remaining NaN values with the mean of their respective columns (if numerical)
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:  # Check if column is numerical
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)  # Fill remaining NaN with mean
        
        # Convert the cleaned DataFrame back to a dictionary
        cleaned_record = df.to_dict(orient='records')[0]
        
        return cleaned_record
    
    except Exception as e:
        print(f"Error while cleaning data: {e}")
        return record  # Return the original record if cleaning fails

# Function to consume messages from the topic and save to a file
def consume_message():
    print("Starting consumer...")
    
    all_cleaned_records = []  # List to store all cleaned records
    
    # Infinite loop to read and process messages from the topic
    for message in consumer:
        raw_record = message.value  # Get the raw message value (deserialized JSON)
        
        print(f"Received raw record: {raw_record}")  # Print raw record
        
        # Clean the received record
        cleaned_record = clean_data(raw_record)
        
        print(f"Cleaned record: {cleaned_record}")  # Print cleaned record
        
        # Append cleaned record to list
        all_cleaned_records.append(cleaned_record)
        
        # Save cleaned data to CSV file incrementally
        save_to_csv(cleaned_record)

def save_to_csv(record):
    """
    Saves a single cleaned record to a CSV file.
    
    Args:
        record (dict): A dictionary representing a single cleaned record.
    """
    try:
        # Convert the record into a DataFrame for saving
        df = pd.DataFrame([record])
        
        # Write headers only if the file does not exist or is empty
        write_header = not os.path.exists(output_file) or os.stat(output_file).st_size == 0
        
        df.to_csv(output_file, mode='a', header=write_header, index=False)
    
    except Exception as e:
        print(f"Error while saving data to CSV: {e}")

if __name__ == '__main__':
    consume_message()  # Start consuming messages
