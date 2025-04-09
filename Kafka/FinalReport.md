### **Introduction**

The rapid urbanization and industrialization of modern cities have led to significant environmental challenges, particularly in monitoring and predicting air quality. High concentrations of pollutants such as carbon monoxide (CO), nitrogen oxides (NOx), and benzene (C6H6) pose severe risks to public health, necessitating real-time monitoring systems to provide timely alerts and actionable insights. This assignment explores the intersection of environmental monitoring and data engineering by leveraging Apache Kafka, a powerful distributed event-streaming platform, to analyze real-time air quality data.


The primary objective of this assignment is to develop a robust pipeline for streaming, processing, and analyzing environmental time series data using the UCI Air Quality dataset. By implementing predictive models for pollutant concentrations, this project aims to demonstrate the practical applications of Kafka in real-time environmental monitoring and decision-making contexts.


Through this hands-on experience, I gained foundational knowledge of Apache Kafka's installation, configuration, and usage for real-time data streaming. Additionally, I performed exploratory data analysis (EDA) on the UCI Air Quality dataset to identify temporal patterns in pollutant concentrations and implement predictive models to forecast air quality metrics. This assignment highlights the critical role of big data techniques in addressing urban environmental challenges, optimizing traffic flow, reducing emissions, and informing policy decisions.

### **Kafka Setup Description**

Apache Kafka is a distributed event-streaming platform designed for high-throughput, fault-tolerant, and real-time data processing. Setting up Kafka involves several steps to ensure its proper installation, configuration, and functioning. Below is a detailed description of the Kafka setup process:

1. Prerequisites: Java Installation: Kafka requires Java to run

2. Downloading Apache Kafka: Downloaded the latest stable binary release from the official [Apache Kafka download page](https://kafka.apache.org/downloads)

3. Configuring Apache Kafka

- Navigate to the extracted Kafka directory
- Modify configuration files as needed

4. Start the Kafka server

6. Create a Topic: Kafka topics are used to store events/messages

7. Start a Producer: A producer sends messages (events) to a Kafka topic

8. Start a Consumer: A consumer reads messages from a Kafka topic. The consumer will display all messages sent by the producer in real time.

Messages are sent from the producer and are received by the consumer.

### Phase 2: Exploratory Data Analysis (EDA) on Time Series Data

**Objective** 

Perform EDA on the air quality data streamed from Kafka, focusing on understanding the temporal patterns in pollutant concentrations and relationships between different pollutants.

**Time Series Analysis**

[Code used to generate the visuals below is here](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase2.ipynb)

For the purpose of this assignment, we focused on  CO, NOx, and Benzene concentrations.

1. Analyze time-based patterns in the data

Missing values were purposely not imputed at this stage as right now, the visualization clearly highlight times when there were blips in the system (when measures are missing). Frequent blips could highlight the need to upgrade infrastructure.

_I visualized hourly pollution concentrations to analyze patterns over time_

- As the visual below show, around Nov-Dec 2004, pollutant level were unusually high. It would be interesting to connect it to events at the time (like earthquakes, waste dumping, air flow patterns) that could have contributed to this
  
   ![newplot](https://github.com/user-attachments/assets/132174c7-defe-441c-8aaf-d7d6d73dc021)

_I also visualized the average hourly and day-of-the-week levels to see if there's a pattern_   

- As the visuals below show, levels are highest in the mornings around 9 AM and then in the evenings, around 7 PM. It could be related to working hours and the associated traffic. If it is, there is room for implementing policies to reduce emissions such as car pooling incentives or offering work from home options. Specific policy recommendation will require deeper dive to identify the causal factors. 

  ![newplot (1)](https://github.com/user-attachments/assets/e5d25a59-eb8a-46ff-a8c9-4f7051af1c03)

- Pollution levels are lowest on the weekends, again suggesting that pollution could be related to work-commute traffic
  
  ![newplot](https://github.com/user-attachments/assets/e37cfa75-da5c-4f0d-97cb-5862a8f6cf66)

2. I also explored the relationship between different pollutants. This could inform variable selection in the modeling phase.

As the visual below shows, CO levels are highly correlated with benzene and NO levels

   ![image](https://github.com/user-attachments/assets/ffc61629-3aa8-4af1-b276-1b81e03ee35d)

3. The Autocorrelation and Partial Autocorrelation plots also inform the modelling stage

As visuals below show, CO levels are most influenced by last 2 values. 

![newplot](https://github.com/user-attachments/assets/1d072a1f-5cc2-4470-9617-d96ade31058d)

The partial autocorrelation plot shows the direct effect of lagged values.

![newplot (1)](https://github.com/user-attachments/assets/3586883c-25c0-4f34-8acd-3fe8d4de9f86)

The above 2 plots suggest that lagged values of degree 2 should be included when developing prediction models for CO. This will become clearer when we compare outcomes of basic modeling with real time modeling. The latter is able to predict much better as the highly relevant data of last 2 hours' CO levels are available here.


### Phase 3: Air Quality Prediction Model

**Objective** 

Develop predictive models to forecast pollutant concentrations (done for CO) using the features derived from the time-series data.

**Feature Engineering**


**Modelling**


_Citations_:

Text (Intro and Kafka setup section are word-for-word from AI) is generated using Perplexity: https://www.perplexity.ai/search/what-error-in-this-line-mean-b-7LHNrTq8Q8OpxFerf90PGw?124=d&125=d&utm_source=copy_output_

[1] https://arxiv.org/abs/2104.01082

[2] https://www.ibm.com/products/instana/supported-technologies/apache-kafka-observability

[3] https://www.logicmonitor.com/blog/what-is-apache-kafka-and-how-do-you-monitor-it

[4] https://middleware.io/blog/kafka-monitoring/

[5] https://www.getorchestra.io/guides/apache-kafka-monitoring-and-metering

[6] https://docs.confluent.io/platform/current/kafka/monitoring.html

[7] https://dl.acm.org/doi/10.1145/3445945.3445949

[8] https://www.datacamp.com/tutorial/apache-kafka-for-beginners-a-comprehensive-guide

[9] https://bell-sw.com/blog/how-to-install-apache-kafka-on-a-local-machine/

[10] https://www.tutorialspoint.com/apache_kafka/apache_kafka_installation_steps.htm

[11] https://bryteflow.com/what-is-apache-kafka-and-installing-kafka-step-by-step/

[12] https://kafka.apache.org/quickstart

[13] https://docs.confluent.io/kafka/introduction.html

[14] https://kafka.apache.org/documentation/

[15] https://www.youtube.com/watch?v=QkdkLdMBuL0

_______________

1. XGBoost
2. ARIMA

   
Model Requirements:
Choose ONE from each category:

Basic Models (Required):
Linear Regression with time-based features
Random Forest
XGBoost
Advanced Models (Optional - 5 Bonus Points):
ARIMA or SARIMA
LSTM (Note: This requires more computational resources)
Feature Engineering Requirements:

Develop time-based features (hour, day, month)
Create lagged features from previous time periods
Generate rolling statistics (averages, standard deviations)
Document your feature engineering approach
Evaluation Process:

Use a chronological train/test split appropriate for time series data
Evaluate using MAE and RMSE metrics
Compare your model to a baseline (previous value prediction)
Integration with Kafka:

Develop a mechanism to use your trained model with incoming Kafka messages
Document how your system would operate in a real-time environment
