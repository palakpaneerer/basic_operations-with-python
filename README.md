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
- `df.drop(columns=['id'])` <- Remove unnecessary columns

# 03. Rewrite Data
- Correctly Update Specific Values in Pandas
When working with pandas DataFrames, attempting to update values through slices can raise a SettingWithCopyWarning. This happens because the operation might modify a copy rather than the original DataFrame, leading to unexpected behavior.  
Incorrect Approach: `df[df['name']=='Mike']['age'] = 30`  
Correct Approach: `df.loc[df['name']=='Mike', 'age'] = 30`  

