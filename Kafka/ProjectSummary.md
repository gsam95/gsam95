This mini-project was done as part of my Operationalizing AI class with Professor Anand Rao at Carnegie Mellon University.


**Objective**

Get hands-on experience with Apache Kafka for real-time data streaming and utilizing it for model development and analysis.


**Context**

As urbanization accelerates, monitoring and predicting air quality has become increasingly critical for public health management and urban planning. High concentrations of air pollutants like CO, NOx, and Benzene can significantly impact respiratory health and overall quality of life. Real-time air quality data analysis is essential for providing timely air quality alerts, optimizing traffic flow to reduce emissions, and informing policy decisions.

I use [UCI Air Quality dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality).

_Dataset Description_

- CSV file format with 9,358 hourly instances (March 2004 to February 2005)
- 15 columns including date/time and various sensor readings
- Missing values are marked with -200 in the dataset
- Features include CO, NOx, NO2, Benzene, and other pollutant measurements
- Ground truth measurements from certified analyzers included alongside sensor readings

Legend - CO: Carbon monoxide, measured in mg/m³ | NOx: Nitrogen oxides, measured in ppb (parts per billion) | NO2: Nitrogen dioxide, measured in µg/m³ | Benzene: Measured in µg/m³ | Normal urban ranges: CO (0.5-5 mg/m³), NOx (5-100 ppb), Benzene (0.5-10 µg/m³)


**Project Summary**

1. [Kafka Setup & Streaming Data](https://github.com/gsam95/gsam95/tree/main/Kafka/Phase1)

- Apache Kafka Setup: Installed Apache Kafka and its dependencies in development environment, configure Kafka servers, and created a Kafka topic for air quality data
- [Kafka producer](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase1/producer.py): Created a Kafka producer script that sends the dataset records to Kafka topic. Simulated real-time data by implementing a time delay mechanism
- [Kafka consumer](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase1/consumer.py): Developed Python script creates a Kafka consumer that read from air quality data topic and processes the incoming data or stores it for analysis
- [Data preprocessing decisions](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase1/datapreprocessingdecision.md) documented here


2. []()








