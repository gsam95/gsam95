## **Data Preprocessing Decisions**

Missing data flows as -200 in the data set. Below is a summary of the decisions made to process the data.

The steps could have been implemented at the consumer stage, but a conscious decision was made to retain original data till visualization stage to identify times when sensors were down. 

### _Data Cleaning Steps_

1. Numeric Columns are identified
   
The columns CO(GT), NOx(GT), and C6H6(GT) are identified as numeric columns that require cleaning.

These columns contain pollutant concentration values.

2. Convert Columns to Float Type
   
Each numeric column is explicitly converted to the float data type to ensure consistent handling of missing values (NaN) and invalid entries.

This step ensures that operations like replacing missing values or calculating column means can be performed without errors.

3. Replace Infinite Values
   
Any occurrences of infinite values (float('inf') or float('-inf')) in the numeric columns are replaced with NaN.

Infinite values can arise due to errors in data collection or calculations, and replacing them ensures the dataset remains clean and usable.

4. Handle Missing Values
   
Missing values in the numeric columns are handled using multiple strategies:

- Forward Fill: Missing values are filled using the value from the previous row (if available). This assumes that nearby data points are similar.
- Backward Fill: Remaining missing values are filled using the value from the next row (if available). This ensures no gaps remain in the dataset.
- Fill with Column Mean: Any remaining missing values after forward and backward filling are replaced with the mean of the respective column. This provides a statistical approximation for missing data.

5. Convert Datetime Column
   
The Datetime column is converted to a proper datetime format using pd.to_datetime(). Invalid datetime entries are coerced into NaT (Not a Time).

Rows with invalid or missing datetime values (NaT) are dropped from the dataset to ensure clean indexing and time-based analysis.





_Reference_

_[Perplexity used to create this steps](https://www.perplexity.ai/search/what-error-in-this-line-mean-b-tLbCAenpTJS59ZRnuLr5eQ)_
