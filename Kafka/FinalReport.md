### Phase 2: Exploratory Data Analysis (EDA) on Time Series Data

**Objective** 

Perform EDA on the air quality data streamed from Kafka, focusing on understanding the temporal patterns in pollutant concentrations and relationships between different pollutants.

**Time Series Analysis**

[Code used to generate the visuals below is here]()

For the purpose of this assignment, we focused on  CO, NOx, and Benzene concentrations.

1. Analyze time-based patterns in the data

Missing values were purposely not imputed at this stage as right now, the visualization clearly highlight times when there were blips in the systems (when measures are missing). Frequent blips could highlight the need to upgrade infrastructure.

_I visualized hourly pollution concentrations to analyze patterns over time_

- As the visual below show, around Nov-Dec 2004, pollutant level were unusually high. It would be interesting to connect it to events at the time (earthquakes, waste dumping, air flow patterns) that could have contributed to this
  
   ![newplot](https://github.com/user-attachments/assets/132174c7-defe-441c-8aaf-d7d6d73dc021)

_I also visualized the average hourly and day-of-the-week levels to see if there's a pattern_   

- As the visuals below show, levels are highest in the mornings around 9 AM and then in the evenings, around 7 PM. It could be related to working hours and the associated traffic. If it is, there is room for implementing policies to reduce emissions such as car pooling incentives or offering work from home options. Specific policy recommendation will require deeper dive to identify the causal factors. 

  ![newplot (1)](https://github.com/user-attachments/assets/e5d25a59-eb8a-46ff-a8c9-4f7051af1c03)

- Pollution levels are lowest on the weekends, again suggesting that pollution could be related to work-commute traffic
  
  ![newplot](https://github.com/user-attachments/assets/e37cfa75-da5c-4f0d-97cb-5862a8f6cf66)

2. I also explored the relationship between different pollutants. This could inform variable selection in the modeling phase.
   
3. Identify seasonality, trends, and anomalies




____

Deliverables:

Basic Visualizations (15 Points):
Time-series plots of
Daily/weekly patterns (average by hour of day, day of week)
Correlation heatmap between different pollutants
Advanced Visualizations (Optional - 5 Bonus Points):
Autocorrelation and partial autocorrelation plots
Decomposition of time series into trend, seasonality, and residuals
Analysis Report:
2-3 page report describing patterns you observed
Analysis of potential factors influencing air quality variations
Discussion of how your findings will inform your modeling approach
