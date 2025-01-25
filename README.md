# basic_operations-with-python
Basic Operations List with Python (Pandas, Numpy, Matplotlyb.pyplot, seaborn)


# 01. Load Data
- tsv file:
  `pd.read_csv('fine_name.tsv, sep='\t')`


# 02. Check Data and Sort Data
- `df.shape` -> (Number of Rows, Number of Columns)
- `df.shape[0]` -> Number of Rows
- `df.shape[1]` -> Number of Columns
- `df.info`
- `df = df.drop(columns=['id'])` <- Remove unnecessary columns
- `df.describe()`
- `df['data'].median()`
- `type(data)`
- `df.dtypes`


# 03. Rewrite Data
- Correctly Update Specific Values in Pandas
When working with pandas DataFrames, attempting to update values through slices can raise a SettingWithCopyWarning. This happens because the operation might modify a copy rather than the original DataFrame, leading to unexpected behavior.  
Incorrect Approach: `df[df['name']=='Mike']['age'] = 30`  
Correct Approach: `df.loc[df['name']=='Mike', 'age'] = 30`  


# 04. Handling Missing Data
### Explicit Missing Data (e.g., NaN, ' ')
- `df.dropna()` <- Rows are considered invalid data and are removed the entire rows.

### Ambiguous Missing Data (e.g., ?, -)
- `df = df.fillna(-999)` <- Explicitly represent missing values with a specific value.
- `df = df.fillna(ffill)`
- `df = df.fillna(bfill)`
- `df = df.fillna(interpolate)`
- `df = df.fillna(df.mean())` <- Temporary solution, using the mean to fill missing values (not generally recommended).
- `df = df.fillna(df.median())` <- (not generally recommended)
- `df = df.fillna(df.mode())` <- (not generally recommended)  
**For a single column:** `df['Nationality'] = df['Nationality'].fillna(df['Nationality'].mode())`
- `df['height'] = df['height'].fillna(df.groupby('Nationality')['height'].transform('median'))` -> Fills missing values in height based on the median for each Nationality.


# 05. Exploratory Data Analysis (EDA)
### Basic Statistics (Max, Min, Mode, Mean, Median, Quartiles, Variance, Standard Deviation (σ))
**Variance:**  
Measures variability, but since it is squared, it changes the unit, making it hard to compare with other statistics.  
`σ² = (1 / N) * Σ(xᵢ - μ)²`  
- σ²: Population variance
- xᵢ: Each data point
- μ: Population mean
- N: Total number of data points

**Standard Deviation (σ):**  
Measures variability in the same unit as the data, making it easier to compare with other statistics.  
`σ = √[(1 / N) * Σ(xᵢ - μ)²]`  
- σ: Population standard deviation 
- xᵢ: Each data point 
- μ: Population mean
- N: Total number of data points

### Visualisation
**Quantitative Data -> Histogram**  
Key aspects to check:
- Normal distribution (e.g., the distribution of male heights)
- Skewed distribution (e.g., income distribution)
- Bimodal distribution (e.g., two different groups like male and female)
- Outliers  
**Code**  
Series_data.plot.hist(title='The desired title for the graph')  
`height_var = df['height']`  
`height_var.plot.hist(title='height', bins=10)`  
`plt.show()`
 
**Qualitative Data: -> Bar Graph**  
Key aspects to check:
- Values are evenly distributed
- One specific value is significantly more frequent than others
- Bar Graph for Qualitative Data:  
**Code**  
Series_data.value_counts()  
`blood_type_var = df['blood_type']`  
`counts = blood_type_var.value_counts()`  
`counts.plot.bar(title='Frequency of blood types')`  
`plt.xlabel('blood types')`  
`plt.ylabel('count')`  
`plt.show()`

# 06. Check Correlation  
### Quantitative Data x Quantitative Data  
**1. Overall Checking**  
`corr_matrix = df.corr()`  
`sns.heatmap(corr_matrix)`  
`plt.show()`  
**2. Detailed Checking**  
plt.scatter(data to plot on x-axis, data to plot on y-axis)  
`plt.scatter(data['weight'],data['height'])`  
`plt.xlabel('weight')`  
`plt.ylabel('height')`  
`plt.show()`  

### Quantitative Data x Qualitative Data
Correlation coefficients cannot be calculated for this type of data.  
sns.boxplot(x='column_name', y='column_name', data=df)  
`sns.boxplot(x='blood_type', y='height', data=df)`  
`plt.show()`  
