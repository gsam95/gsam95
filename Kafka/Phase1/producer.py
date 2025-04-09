import pandas as pd
import time
from kafka import KafkaProducer
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("kafka_producer.log"),
        logging.StreamHandler()
    ]
)

# Kafka producer configuration
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],  # Replace with your Kafka broker address
    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')  # Use `default=str` for non-serializable objects
)

# Function to clean the dataset
def clean_dataset(file_path):
    """
    Reads and cleans the dataset.
    
    Args:
        file_path (str): Path to the dataset CSV file.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        # Read the dataset
        df = pd.read_csv(file_path, sep=';', decimal=',', na_values=-200)
        
        # Drop unnecessary columns (e.g., empty columns)
        #df = df.dropna(axis=1, how='all')
        
        # Combine Date and Time into a single datetime column
        #df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
        
        # Drop original Date and Time columns
        #df = df.drop(['Date', 'Time'], axis=1)
        
        # Sort by datetime just in case
        #df = df.sort_values(by='Datetime').reset_index(drop=True)
        
        return df
    except Exception as e:
        logging.error(f"Error while cleaning dataset: {e}")
        raise

# Function to send records to Kafka topic
def send_to_kafka(df, topic):
    """
    Sends records from the DataFrame to a Kafka topic with a delay.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        topic (str): Kafka topic name.
    """
    try:
        for index, row in df.iterrows():
            # Convert row to dictionary for JSON serialization and handle NaT values
            record = row.to_dict()
            if pd.isnull(record['Datetime']):  # Check for NaT
                record['Datetime'] = "Unknown"  # Replace NaT with a default value
            else:
                record['Datetime'] = record['Datetime'].strftime('%Y-%m-%d %H:%M:%S')  # Format valid timestamps
            
            # Send record to Kafka topic
            producer.send(topic, record)
            logging.info(f"Sent record {index + 1}/{len(df)}: {record}")
            
            # Simulate real-time delay for hourly readings (1 second here for testing; adjust as needed)
            time.sleep(0.1) #assuming this is equivalent to hour for this assignment # Microsecond delay (use `time.sleep(3600)` for actual hourly simulation)
    
    except Exception as e:
        logging.error(f"Error while sending data to Kafka: {e}")
    
    finally:
        # Ensure all messages are sent before exiting
        producer.flush()
        producer.close()
        logging.info("Kafka producer closed.")

# Main function
if __name__ == '__main__':
    try:
        # Path to the dataset CSV file
        file_path = "D:/Grace/CMU/Courses/Spring2025/OpAI/Assignment/1/AirQualityUCI.csv"  # Replace with your actual file path
        
        # Clean the dataset
        cleaned_data = clean_dataset(file_path)
        
        # Define Kafka topic name
        kafka_topic = "test-topic"
        
        # Send cleaned data to Kafka topic
        send_to_kafka(cleaned_data, kafka_topic)
    
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
