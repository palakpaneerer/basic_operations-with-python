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

**Other settings**
- `plt.figure(figsize=(5,5))`


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
`y = df['height']`  
`X = df[['weight', 'blood_type', 'nationality']`  
`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32, test_size=0.2)` <-Default of test_size: 0.25  
`X_train_line = X_train.shape[0]`  
`X_test_line = X_test.shape[0]`  
`print(f'X_train_line: {X_train_line} rows, X_test_line: {X_test_line} rows')`  


# 08. Make a Model
**Algorithms Summary**
| Category           | Algorithms                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------|
| Linear Regression  | Multiple Regression, Ridge, Lasso, ElasticNet, Logistic Regression                        |
| Tree-Based Models  | Decision Tree, Random Forest, GBDT, LightGBM                                              |
| Neural Networks    | MLP (Multi-Layer Perceptron), RNN (Recurrent Neural Network), CNN (Convolutional Neural Network) |
| Time Series Models | AR (Autoregressive), MA (Moving Average), ARIMA (Autoregressive Integrated Moving Average) |
| Others             | Naive Bayes, SVM (Support Vector Machine)                                                 |

**Multiple regression model**  
`from sklearn.linear_model import LinearRegression as LR`  
`lr = LR()`  
`lr.fit(X_train, y_train)`  


# 09. Predict
`y_pred_train = lr.predict(X_train)`  
`y_pred_test = lr.predict(X_test)`  
`print(y_pred_train)`
`print(y_pred_test)`  


# 10. Validate
**Metrics Summary**
| Task               | Evaluation Metric | Range     | Interpretation         | Features                                   |
|--------------------|-------------------|-----------|------------------------|-------------------------------------------|
| Regression Task    | MAE               | 0 to ∞    | Smaller is better      | Easy to interpret                         |
|                    | RMSE              | 0 to ∞    | Smaller is better      | Penalizes large errors significantly (i.e., creates a model that avoids large prediction errors) |
| Classification Task | Accuracy         | 0 to 1    | Larger is better       | Easy to interpret                         |
|                    | AUC               | 0 to 1    | Larger is better       | Useful for binary classification problems |
|                    | Log Loss          | 0 to ∞    | Smaller is better      | Applicable for multi-class classification |

**RMSE: sklearn doesn’t provide RMSE directly, so calculate it by taking the square root of MSE.**   
`from sklearn.metrics import mean_squared_error as MSE`  
- Calucalate MSE  
`mse_train = MSE(y_train, y_pred_train)`: MSE(Actual data, Predicted Data)  
`mse_test = MSE(y_test, y_pred_test)`  
- Calculate RMSE  
`rmse_train = np.sqrt(mse_train)`  
`rmse_test = np.sqrt(mse_test)`  
`print(rmse_train)`
`print(rmse_test)`  


# 11. Visualise the validation
To evaluate how well the predicted values match the actual values, you can use a scatter plot. Points along the diagonal line indicate a good match.  
`plt.scatter(y_test, y_pred)`: plt.scatter(x-axis values, y-axis values)  
- Get the minimum and maximum values of y_test and y_pred  
`y_test_min = np.min(y_test)`  
`y_test_max = np.max(y_test)`  
`pred_min = np.min(y_pred)`  
`pred_max = np.max(y_pred)`  
- Compare the values to determine the final minimum and maximum values  
`min_value = np.minimum(y_test_min, pred_min)`  
`max_value = np.maximum(y_test_max, pred_max)`  
`print(min_value, max_value)`  
- Set the range for the x-axis and y-axis  
`plt.xlim([min_value, max_value])`  
`plt.ylim([min_value, max_value])`  
- Draw a diagonal line  
`plt.plot([min_value, max_value], [min_value, max_value])`  
- Add labels to the x-axis and y-axis  
`plt.xlabel('Actual Values')`  
`plt.ylabel('Predicted Values')`  
- Show the plot  
`plt.show()`  


---
# C. Step 3: Improving the Model
# 12. Dummy variables for categorical data  
- Import  
`from sklearn.model_selection import train_test_split`  
`from sklearn.linear_model import LinearRegression as LR`  
`from sklearn.metrics import mean_squared_error as MSE`  
- Prepare variables for the target and explanatory variables  
`y = df['height']`  
`X = df[['weight', 'blood_type', 'nationality']]`  
- **Get dummies**  
**`X = pd.get_dummies(X)` <- Convert all categorical explanatory variables into dummy variables**  
- Split the data  
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)  
- Make a model  
`lr = LR()`  
`lr.fit(X_train, y_train)`  
- Predict  
`y_pred_train = lr.predict(X_train)`   
`y_pred_test = lr.predict(X_test)`  
- Calculate MSE (Mean Squared Error)  
`mse_train = MSE(y_train, y_pred_train)`  
`mse_test = MSE(y_test, y_pred_test)`  
- Calculate RMSE (Root Mean Squared Error)  
`rmse_train = np.sqrt(mse_train)`  
`rmse_test = np.sqrt(mse_test)`  
`print(rmse_train)`  
`print(rmse_test)`  


# 13. Log Transformation
Log transformation is primarily used in linear models (e.g., linear regression, multiple regression) to linearize features that exhibit exponential growth or decay, making them easier to model and improving prediction accuracy. While it is commonly applied in linear models, non-linear models (e.g., random forests or neural networks) can inherently capture complex non-linear relationships, so log transformation is not always necessary for them.
- Import  
`from sklearn.model_selection import train_test_split`  
`from sklearn.linear_model import LinearRegression as LR`  
`from sklearn.metrics import mean_squared_error as MSE`  
- **Log Transformation**   
`df['salary_log']= np.log(df['salary'])`  
- Prepare variables for the target and explanatory variables  
`y = df['height']`  
`X = df[['weight', 'blood_type', 'nationality', 'salary_log']]`  
- Split data  
`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)`  
- Make a model  
`lr = LR()`  
`lr.fit(X_train, y_train)`  
- Predict  
`y_pred_train = lr.predict(X_train)`  
`y_pred_test = lr.predict(X_test)`
- Calculate MSE (Mean Squared Error)  
`mse_train = MSE(y_train, y_pred_train)`  
`mse_test = MSE(y_test, y_pred_test)`  
- Calculate RMSE (Root Mean Squared Error)  
`rmse_train = np.sqrt(mse_train)`  
`rmse_test = np.sqrt(mse_test)`  
`print(rmse_train)`  
`print(rmse_test)`  
