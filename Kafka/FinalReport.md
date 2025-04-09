### Phase 2: Exploratory Data Analysis (EDA) on Time Series Data

**Objective** 

Perform EDA on the air quality data streamed from Kafka, focusing on understanding the temporal patterns in pollutant concentrations and relationships between different pollutants.

**Time Series Analysis**

[Code used to generate the visuals below is here](https://github.com/gsam95/gsam95/blob/main/Kafka/Phase2.ipynb)

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

As the visual below shows, CO levels are highly correlated with benzene and NO levels

   ![image](https://github.com/user-attachments/assets/ffc61629-3aa8-4af1-b276-1b81e03ee35d)

3. The Autocorrelation and Partial Autocorrelation plots can inform the modelling stage.

As visuals below show, CO levels are most influenced by last 2 values. 

![newplot](https://github.com/user-attachments/assets/1d072a1f-5cc2-4470-9617-d96ade31058d)

The partial autocorrelation plot shows the direct effect of lagged values.

![newplot (1)](https://github.com/user-attachments/assets/3586883c-25c0-4f34-8acd-3fe8d4de9f86)

The above 2 plots suggest that lagged values of degree 2 should be included when developing prediction models for CO. This will become clearer when we compare outcomes of basic modeling with real time modeling. The latter is able to predict much better as the highly relevant data of last 2 hours' CO levels are available here.


____

Deliverables:

Advanced Visualizations (Optional - 5 Bonus Points):

Decomposition of time series into trend, seasonality, and residuals
Analysis Report:
2-3 page report describing patterns you observed
Analysis of potential factors influencing air quality variations
Discussion of how your findings will inform your modeling approach
