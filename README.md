# basic_operations-with-python
Basic Operations List with Python (Pandas, Numpy, Matplotlyb.pyplot, seaborn)

---
# A. Step 1: Preprocessing
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
**For a single column:** `df['nationality'] = df['nationality'].fillna(df['nationality'].mode())`
- `df['height'] = df['height'].fillna(df.groupby('nationality')['height'].transform('median'))` -> Fills missing values in height based on the median for each Nationality.


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


---
# B. Step 2: Making Models
# 07. Split Data
`from sklearn.model_selection import train_test_split`
`y = data['height']`
`X = data[['weight', 'blood_type', 'nationality']`
`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32, test_size=0.2)` <-Default of test_size: 0.25
`X_train_line = X_train.shape[0]`
`X_test_line = X_test.shape[0]`
`print(f'X_train_line: {X_train_line} rows, X_test_line: {X_test_line} rows')`



# 08. Set Evaluation
**RMSE: sklearn doesn’t provide RMSE directly, so calculate it by taking the square root of MSE.**  
`from sklearn.metrics import mean_squared_error as MSE`  
`actual = [3,4,6,2,4,6,1]`  
`pred = [4,2,6,5,3,2,3]`  
`mse = MSE(actual, pred)`  <- MSE(Actual values, Predicted values)
`rmse = np.sqrt(mse)`  
`print(rmse)`  




**Algorithms Summary**
| Category           | Algorithms                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------|
| Linear Regression  | Multiple Regression, Ridge, Lasso, ElasticNet, Logistic Regression                        |
| Tree-Based Models  | Decision Tree, Random Forest, GBDT, LightGBM                                              |
| Neural Networks    | MLP (Multi-Layer Perceptron), RNN (Recurrent Neural Network), CNN (Convolutional Neural Network) |
| Time Series Models | AR (Autoregressive), MA (Moving Average), ARIMA (Autoregressive Integrated Moving Average) |
| Others             | Naive Bayes, SVM (Support Vector Machine)                                                 |



# 08. Evaluation Metrics
**Metrics Summary**
| Task               | Evaluation Metric | Range     | Interpretation         | Features                                   |
|--------------------|-------------------|-----------|------------------------|-------------------------------------------|
| Regression Task    | MAE               | 0 to ∞    | Smaller is better      | Easy to interpret                         |
|                    | RMSE              | 0 to ∞    | Smaller is better      | Penalizes large errors significantly (i.e., creates a model that avoids large prediction errors) |
| Classification Task | Accuracy         | 0 to 1    | Larger is better       | Easy to interpret                         |
|                    | AUC               | 0 to 1    | Larger is better       | Useful for binary classification problems |
|                    | Log Loss          | 0 to ∞    | Smaller is better      | Applicable for multi-class classification |



データ分割

# kplを取り出し、変数yに代入
y = data['kpl']
# 6つのカラムを指定し、説明変数を表す変数Xを作成
X = data[['cylinders','displacement','horsepower','acceleration','model_year','origin']]


# scikit-learnライブラリからtrain_test_split関数をインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32, test_size=0.2) <-test_sizeのデフォルトは0.25

# X_trainとX_testから行数を取り出し、それぞれ変数X_train_line, X_test_lineに代入し、表示
X_train_line = X_train.shape[0]
X_test_line = X_test.shape[0]
print(X_train_line)
print(X_test_line)

