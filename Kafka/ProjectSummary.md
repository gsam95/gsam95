# [**Learning Kafka**](https://github.com/gsam95/gsam95/tree/main/Kafka)

This mini-project was done as part of my Operationalizing AI class with Professor Anand Rao at Carnegie Mellon University.

Below is summary of the context and project phases.



### **Objective**

Get hands-on experience with Apache Kafka for real-time data streaming and utilizing it for model development and analysis.


#### **Context**

As urbanization accelerates, monitoring and predicting air quality has become increasingly critical for public health management and urban planning. High concentrations of air pollutants like CO, NOx, and Benzene can significantly impact respiratory health and overall quality of life. Real-time air quality data analysis is essential for providing timely air quality alerts, optimizing traffic flow to reduce emissions, and informing policy decisions.

I use [UCI Air Quality dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality).

_Dataset Description_

- CSV file format with 9,358 hourly instances (March 2004 to February 2005)
- 15 columns including date/time and various sensor readings
- Missing values are marked with -200 in the dataset
- Features include CO, NOx, NO2, Benzene, and other pollutant measurements
- Ground truth measurements from certified analyzers included alongside sensor readings

Legend - CO: Carbon monoxide, measured in mg/m³ | NOx: Nitrogen oxides, measured in ppb (parts per billion) | NO2: Nitrogen dioxide, measured in µg/m³ | Benzene: Measured in µg/m³ | Normal urban ranges: CO (0.5-5 mg/m³), NOx (5-100 ppb), Benzene (0.5-10 µg/m³)


### **Project Phases - Summary**

#### 1. [Kafka Setup & Streaming Data](https://github.com/gsam95/gsam95/tree/main/Kafka/Phase1)

- [Apache Kafka Setup](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase1/kafkasetup.md): Installed Apache Kafka and its dependencies in development environment, configured Kafka servers, and created a Kafka topic for air quality data
- [Kafka producer](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase1/producer.py): Created a Kafka producer script that sends the dataset records to Kafka topic. Simulated real-time data by implementing a time delay mechanism
- [Kafka consumer](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase1/consumer.py): Developed Python script creates a Kafka consumer that reads from topic and processes the incoming data or stores it for analysis
- [Data preprocessing decisions](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase1/datapreprocessingdecision.md) documented here



_Now that we have real time data streaming in, there are multiple ways to use. In Phase 2 we visualize the streamed data. In Phase 3 we make hourly predictions to guide decisions._


#### 2. *Visualizing Patterns*

Details of the EDA on the air quality data streamed from Kafka, focusing on understanding the temporal patterns in pollutant concentrations and relationships between different pollutants are documented in the [Final Report](https://github.com/gsam95/gsam95/blob/main/Kafka/FinalReport.md).


_Future Scope of Work_

_Here we took a consolidated snapshot of the streamed data and visualized it to analyze patterns. A more real-time use case would be to visualize the data hourly (an example of this would be the screens that show pollution levels. In future, I'd like to integrate Kafka and make real time visualizations._

[![image](https://github.com/user-attachments/assets/1b221f7f-4262-4f9f-a882-9e252095248a)](https://www.dreamstime.com/air-pollution-index-api-air-pollution-index-roadsign-electronic-screen-many-uses-environment-pollution-control-image109106661)


#### 3. *Real-time Predictions*



_**Deliverables**_

1. [Final Report](https://github.com/gsam95/gsam95/blob/main/Kafka/FinalReport.md) can be found here. It logs output for each phase
2. Other deliverables required for the assignment are linked in the corresponding section and subsection of this page





