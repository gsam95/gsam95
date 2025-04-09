### **Introduction**

The rapid urbanization and industrialization of modern cities have led to significant environmental challenges, particularly in monitoring and predicting air quality. High concentrations of pollutants such as carbon monoxide (CO), nitrogen oxides (NOx), and benzene (C6H6) pose severe risks to public health, necessitating real-time monitoring systems to provide timely alerts and actionable insights. This assignment explores the intersection of environmental monitoring and data engineering by leveraging Apache Kafka, a powerful distributed event-streaming platform, to analyze real-time air quality data.


The primary objective of this assignment is to develop a robust pipeline for streaming, processing, and analyzing environmental time series data using the UCI Air Quality dataset. By implementing predictive models for pollutant concentrations, this project aims to demonstrate the practical applications of Kafka in real-time environmental monitoring and decision-making contexts.


Through this hands-on experience, I gained foundational knowledge of Apache Kafka's installation, configuration, and usage for real-time data streaming. Additionally, I performed exploratory data analysis (EDA) on the UCI Air Quality dataset to identify temporal patterns in pollutant concentrations and implement predictive models to forecast air quality metrics. This assignment highlights the critical role of big data techniques in addressing urban environmental challenges, optimizing traffic flow, reducing emissions, and informing policy decisions.

### **Kafka Setup Description**

Apache Kafka is a distributed event-streaming platform designed for high-throughput, fault-tolerant, and real-time data processing. Setting up Kafka involves several steps to ensure its proper installation, configuration, and functioning. Below is an overview of the Kafka setup process:

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


**The above 2 plots suggest that lagged values of degree 2 should be included when developing prediction models for CO. This will become clearer when we compare outcomes of basic modeling with real time modeling. The latter is able to predict much better as the highly relevant data of last 2 hours' CO levels are available here.**


_____________________________________________________________________________________________________________________________________________________________


### Phase 3: Air Quality Prediction Model

**Objective** 

Develop predictive models to forecast pollutant concentrations (done for CO) using the features derived from the time-series data.

**Feature Engineering**

The feature engineering process involves creating new variables that capture meaningful patterns and relationships in the data. Below is a description of the features created based on the provided code:


1. Time-Based Features: Hourly, daily, and monthly patterns help capture temporal variations in pollutant levels
   
These features are extracted from the Datetime index to capture temporal patterns in pollutant concentrations:

- Hour: Represents the hour of the day (e.g., 0 to 23). It helps identify hourly variations in air quality.
- Day: Represents the day of the month (e.g., 1 to 31). It can be used to analyze daily trends.
- Month: Represents the month of the year (e.g., 1 for January, 12 for December). This feature is useful for identifying seasonal patterns in pollutant levels.

2. Lagged Features: Lagged features model how past pollutant levels influence current observations

Lagged features represent pollutant concentrations from previous time periods (lags), capturing temporal dependencies:

- CO_lag_1, CO_lag_2, CO_lag_3: Carbon monoxide concentrations from 1, 2, and 3 time periods before the current observation. These features help model how past CO levels influence current levels.
- NOx_lag_1, NOx_lag_2, NOx_lag_3: Nitrogen oxide concentrations from 1, 2, and 3 previous periods. These features capture short-term dependencies in NOx levels.
- C6H6_lag_1, C6H6_lag_2, C6H6_lag_3: Benzene concentrations from 1, 2, and 3 previous periods. These features help model temporal relationships in benzene levels.

3. Rolling Statistics: Rolling statistics smooth out fluctuations and highlight trends or variability over short time windows

Rolling statistics provide aggregated measures (mean and standard deviation) over a defined window of time to capture local trends and variability:

- CO_roll_mean: The average carbon monoxide concentration over a rolling window of three time periods. This feature smooths out short-term fluctuations and highlights trends.
- CO_roll_std: The standard deviation of carbon monoxide concentrations over three time periods. It measures variability in CO levels within the window.
- NOx_roll_mean: The average nitrogen oxide concentration over a rolling window of three time periods. This feature captures local trends in NOx levels.
- NOx_roll_std: The standard deviation of nitrogen oxide concentrations over three time periods. It highlights variability in NOx levels within the window.



_Target Variable_

The target variable for predictive modeling is CO(GT) - Carbon monoxide concentration measured by ground truth sensors. This is the dependent variable that predictive model aims to forecast.


**Modelling**

The model used in this analysis is **XGBoost (Extreme Gradient Boosting)**, a powerful and efficient machine learning algorithm designed for supervised learning tasks. XGBoost is particularly well-suited for regression problems due to its ability to handle complex relationships between features and the target variable, while minimizing overfitting through regularization techniques.


Key characteristics of the XGBoost model:

- Number of Estimators: 100 decision trees are built during training.
- Maximum Depth: Each tree has a maximum depth of 5, allowing the model to capture moderately complex patterns in the data.
- Learning Rate: Set to 0.1, which controls the step size in updating weights during training. This balances convergence speed and model performance.


The model was trained to predict carbon monoxide concentrations (CO(GT)) based on engineered features such as lagged values, rolling statistics, and time-based attributes.

_Chronological Train/Test Split_

Given that this is a time series problem, a chronological train/test split was applied to ensure temporal consistency:

1. Training Data: All observations from the year 2004 were used for training the model. This ensures that predictions are based only on past data
2. Testing Data: All observations from the year 2005 were used for testing. The test set represents future data that the model has not seen during training

This approach respects the sequential nature of time series data and avoids data leakage (e.g., using future information during training).


_Evaluation Metrics_

The model's performance was evaluated using:
1. Mean Absolute Error (MAE):
   
   - Measures the average magnitude of errors between predicted and actual values
   - MAE is easy to interpret; lower values indicate better performance
   - Result: MAE = 0.11, indicating that on average, the model's predictions deviate from actual CO concentrations by 0.11 units

The low MAE value (0.11) indicates that the model provides highly accurate predictions with minimal deviation from actual values.

2. Root Mean Squared Error (RMSE):
   
   - Measures the square root of the average squared errors between predicted and actual values
   - RMSE penalizes larger errors more heavily than MAE, making it sensitive to outliers
   - Result: RMSE = 0.16, suggesting that the typical prediction error is approximately 0.16 units

The RMSE value (0.16) reinforces this conclusion, showing that larger errors are rare and well-controlled.


### **Integration with Kafka**

In a real-time environment:

1. Data Streaming: Environmental data is continuously streamed into Kafka from sensors or other sources
2. Consumption: The Kafka consumer retrieves incoming messages, cleans them, and appends them to a local CSV file
3. Feature Engineering: Historical data is transformed into lagged features and rolling statistics required for predictions
4. Prediction: The trained XGBoost model generates pollutant concentration forecasts for the next hour based on processed features
5. Output: Predictions are saved locally and can be used for real-time monitoring or decision-making


The developed mechanism integrates the trained machine learning model (XGBoost regressor) with a Kafka consumer pipeline to enable real-time predictions of pollutant concentrations. The process involves consuming environmental data streams from Kafka, preprocessing the data, and generating predictions using the trained model. 

Below is a detailed description of how this mechanism was implemented here:

1. Kafka Consumer Pipeline

The system begins by initializing a Kafka consumer to listen to messages from the specified topic (`test-topic`). Each message represents a record of environmental data, such as pollutant concentrations and meteorological parameters. The consumer:

- Connects to the Kafka broker (`localhost:9092`) and retrieves messages in real time
- Deserializes incoming JSON messages into Python dictionaries for further processing

2. Data Cleaning

Once a message is received, the `clean_data()` function processes the record to handle missing or invalid values:

- Invalid entries (e.g., `-200` or `'Unknown'`) are replaced with appropriate placeholders (`NaN`)
- Missing values are handled using forward filling, backward filling, or replacement with column means (if applicable)
- 
This ensures that the incoming data is clean and usable for feature engineering and prediction.

3. Saving Data to CSV

After cleaning, each record is appended to a local CSV file (`streamed_kafka_data.csv`). This file serves as a cumulative log of all received data and provides historical context for generating lagged features and rolling statistics required for predictions.

4. Feature Engineering

The `preprocess_for_prediction()` function transforms the raw data into features suitable for prediction:

- Lagged Features: Creates lagged values (e.g., CO concentrations from 1, 2, and 3 previous hours) to capture temporal dependencies
- Rolling Statistics: Computes rolling averages and standard deviations over a 3-hour window to capture local trends and variability
- The processed data is aligned with the feature list used during model training to ensure consistency between training and prediction phases.

5. Prediction Using Trained Model

Once the features are prepared:

- The last row of processed features is extracted as input for the trained XGBoost model
- The model predicts the pollutant concentration (CO) for the next hour based on historical patterns and trends
- The predicted value is paired with the corresponding datetime (incremented by one hour from the last observed timestamp)

6. Saving Predictions

Predictions are saved to another CSV file (`hourly_predictions.csv`) in real time. Each entry includes:

- The predicted datetime
- The predicted pollutant concentration (CO)

This file provides a record of hourly forecasts generated by the system.


_Benefits of Real-Time Prediction via Kafka_

1. Scalability: Kafka's distributed architecture allows seamless handling of large volumes of streaming data
2. Timeliness: Predictions are generated in near real-time, enabling proactive responses to air quality changes
3. Robustness: The integration of feature engineering ensures that predictions account for temporal dependencies and local trends in pollutant levels





















__

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
