# basic_operations-with-python
Basic Operations List with Python (Pandas, Numpy, Matplotlyb.pyplot)


# 01. Load Data
- tsv file:
  `pd.read_csv('fine_name.tsv, sep='\t')`

# 02. Check Data and Sort Data
- `df.shape` -> (Number of Rows, Number of Columns)
- `df.shape[0]` -> Number of Rows
- `df.shape[1]` -> Number of Columns
- `df.info`
- `df = df.drop(columns=['id'])` <- Remove unnecessary columns

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
