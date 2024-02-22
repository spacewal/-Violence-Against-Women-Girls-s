#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Set the Environment
# Ignore Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
#Write out the versions of all packages to requirements.txt
get_ipython().system('pip freeze >> requirements.txt')
#!pip unfreeze requirements.txt

# Remove the restriction on Jupyter that limits the columns displayed (the ... in the middle)
pd.set_option('display.max_columns',None)
# Docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html#

# Pretty Display of variables.  for instance, you can call df.head() and df.tail() in the same cell and BOTH display w/o print
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# List of ALL Magic Commands.  To run a magic command %var  --- i.e.:  %env
get_ipython().run_line_magic('lsmagic', '')
# %env  -- list environment variables
# %%time  -- gives you information about how long a cel took to run
# %%timeit -- runs a cell 100,000 times and then gives you the average time the cell will take to run (can be LONG)
# %pdb -- python debugger

# to display nice model diagram
from sklearn import set_config
set_config(display='diagram')

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

print(np.__version__)
print(sklearn.__version__)


# In[6]:


get_ipython().run_cell_magic('time', '', '%env\nprint("Hello World")\n')


# In[7]:


# Standard Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')

#Visualization
import matplotlib
import seaborn as sns
import plotly.express as px

#Machine Learming Model
from sklearn.linear_model import LogisticRegression


# # Find and Load a Large Dataset

# In[8]:


df = pd.read_csv("https://raw.githubusercontent.com/spacewal/-Violence-Against-Women-Girls-s/main/womens_data_original.csv")


# # Structural Investigation

# ### First Few Rows:
# The dataset contains columns for `RecordID`, `Country`, `Gender`, `Demographics Question`, `Demographics Response`, `Question`, `Survey Year`, and `Value`. The first few rows show data related to Afghanistan, with demographic questions about marital status and education, and responses to a question about burning food.
# 

# In[9]:


# Display the first few rows of the dataset
df.head()


# In[10]:


# Get the data types of each column
df.dtypes


# In[11]:


# Get the number of unique values in each column
df.nunique()


# ### Missing Values:
# 
# There are no missing values in most of the columns, except for the `Value` column, which has 1,413 missing values.
# 

# In[12]:


# Check for missing values
df.isnull().sum()


# ### Basic Statistics for Numeric Columns:
# 
# - `RecordID`: Ranges from 1 to 420 with a mean of 210.5.
# - `Value`: Ranges from 0 to 86.9 with a mean of 19.76. The distribution of values is right-skewed, as indicated by the mean being greater than the median (14.9).
# 

# In[13]:


# Get basic statistics for each column
df.describe()


# In[14]:


df.info()


# # Quality Investigation

# In[15]:


# 1. Handling Missing Values
# Fill missing values with the mean of the column
df['Value'] = df['Value'].fillna(df['Value'].mean())


# In[16]:


df.dropna(subset=['Value'], inplace=True)


# In[17]:


# 2. Converting Data Types
# Convert 'Survey Year' to datetime
df['Survey Year'] = pd.to_datetime(df['Survey Year'], format='%d/%m/%Y')


# In[18]:


# 3. Removing Duplicates
df.drop_duplicates(inplace=True)


# In[19]:


# 4. Handling Outliers (using Z-score method for the 'Value' column)
z_scores = stats.zscore(df['Value'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)  # Keep rows with Z-score less than 3
df = df[filtered_entries]


# In[20]:


# 5. Normalization/Standardization (Optional)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Value'] = scaler.fit_transform(df[['Value']])


# ### Handling Missing Values:
# 
# Missing values in the `Value` column can be filled with a specific value (e.g., mean, median) or the rows with missing values can be dropped.
# 
# ### Converting Data Types:
# 
# The `Survey Year` column should be converted to datetime type for better analysis and handling.
# 
# ### Removing Duplicates:
# 
# Duplicate rows, if any, should be removed to ensure the uniqueness of each record.
# 
# ### Handling Outliers:
# 
# Outliers in the `Value` column can be identified and treated using methods like the Interquartile Range (IQR) or Z-scores.
# 
# ### Standardization:
# 
# The `Value` column can be standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1, which is useful for certain machine learning algorithms.
# 

# # Merge with a Second DataFrame

# In[21]:


df_pop = pd.read_csv("https://raw.githubusercontent.com/spacewal/-Violence-Against-Women-Girls-s/main/population-and-demography.csv")


# In[19]:


df_pop.head()


# In[20]:


# Rename columns to ensure consistency
df_pop = df_pop.rename(columns={'Country name': 'Country'})
df = df.rename(columns={'Survey Year': 'Year'})


# In[21]:


# Convert 'Year' columns to the same format (e.g., to datetime or to integer)
df['Year'] = pd.to_datetime(df['Year'], format='%d/%m/%Y').dt.year
df_pop['Year'] = pd.to_datetime(df_pop['Year'], format='%Y').dt.year


# In[22]:


# Merge the datasets on the 'Country' and 'Year' columns, selecting only 'Population' from the population data
merged_data = pd.merge(df, df_pop[['Country', 'Year', 'Population']], on=['Country', 'Year'], how='inner')


# In[23]:


merged_data.head()


# ### Loading the Datasets:
# - We start by loading both the women's data and the population data into Pandas DataFrames using the `pd.read_csv()` function.
# 
# ### Renaming Columns:
# - To ensure consistency between the two datasets, we rename the relevant columns so that they match. In this case, we rename the 'Survey Year' column in the women's data to 'Year' and the 'Country name' column in the population data to 'Country' using the `rename()` method.
# 
# ### Converting 'Year' Columns:
# - We convert the 'Year' columns in both datasets to the same format. In this example, we convert them to integer years using the `pd.to_datetime()` function followed by the `.dt.year` attribute. This step is crucial for ensuring that the merge can be performed correctly based on the 'Year' column.
# 
# ### Merging the Datasets:
# - We perform the merge using the `pd.merge()` function. We specify the columns to merge on by setting `on=['Country', 'Year']`. This means that the merge will be based on matching values in both the 'Country' and 'Year' columns of the two datasets.
# - We use an inner join (`how='inner'`) for the merge, which means that only rows with matching 'Country' and 'Year' values in both datasets will be included in the merged dataset.
# - We select only the 'Country', 'Year', and 'Population' columns from the population data to be included in the merged dataset by using `df_pop[['Country', 'Year', 'Population']]`.
# 
# ### Displaying the Merged Dataset:
# - Finally, we display the first few rows of the merged dataset using the `head()` method to verify that the merge was performed correctly.
# 

# # Data Binning

# In[24]:


# Define the bin edges and labels
bin_edges = [merged_data['Value'].min(), merged_data['Value'].quantile(0.33), merged_data['Value'].quantile(0.66), merged_data['Value'].max()]
bin_labels = ['Low', 'Medium', 'High']

# Bin the 'Value' column
merged_data['Value_Binned'] = pd.cut(merged_data['Value'], bins=bin_edges, labels=bin_labels)

# Display the first few rows of the dataset with the binned column
print(merged_data.head())


# In[25]:


merged_data.head()


# ### Define the Bin Edges and Labels:
# - I define the edges of the bins using the minimum value, the 33rd percentile (quantile 0.33), the 66th percentile (quantile 0.66), and the maximum value of the 'Value' column. This creates three bins with roughly equal numbers of data points, which is useful for handling skewed distributions.
# - I also define the labels for each bin as 'Low', 'Medium', and 'High', which will be used to categorize the values.
# 
# ### Bin the 'Value' Column:
# - I use the `pd.cut()` function to bin the 'Value' column into the defined bins. The `bins` parameter specifies the edges of the bins, and the `labels` parameter assigns the corresponding labels to each bin.
# 
# ### Display the Dataset:
# - I print the first few rows of the dataset with the new 'Value_Binned' column to verify that the binning has been applied correctly.
# 

# In[26]:


merged_data.info()


# # Feature Engineering

# In[27]:


merged_data.info()


# In[28]:


from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler instance
min_max_scaler = MinMaxScaler()

# Normalize the 'Population' column
merged_data['Normalized_Population'] = min_max_scaler.fit_transform(merged_data[['Population']].values)


# In[29]:


merged_data.head()


# # Lambda Function Application

# In this example, the lambda function calculates the "Value_Per_Capita" by dividing the "Value" column by the "Population" column for each row. The `apply` method is used to apply this lambda function row-wise across the DataFrame. The `axis=1` argument specifies that the function should be applied along the rows.
# 

# In[30]:


# Lambda

merged_data['Value_Per_Capita'] = merged_data.apply(lambda row: row['Value'] / row['Population'] if row['Population'] != 0 else 0, axis=1)


# In[31]:


merged_data.head()


# # Deep Exploratory Data Analysis

# In[32]:


merged_data.describe().T


# In[34]:


merged_data = merged_data.dropna(subset=['Value_Binned'])


# In[36]:


# Checking for Null Values
merged_data.isnull().sum()


# In[38]:


merged_data.nunique()


# In[39]:


print('Gender ' + str(sorted(merged_data['Gender'].unique())))


# In[40]:


print('Demographics Question ' + str(sorted(merged_data['Demographics Question'].unique())))


# In[41]:


print('Demographics Response ' + str(sorted(merged_data['Demographics Response'].unique())))


# In[42]:


get_ipython().system('pip install ydata_profiling')


# In[43]:


from ydata_profiling import ProfileReport


# In[44]:


profile = ProfileReport(merged_data, title="Profiling Report")


# In[45]:


profile


# # Feature Importance Analysis

# In[48]:


print(merged_data.dtypes)


# In[49]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()


# In[62]:


numerics = ['int64','float64']
catDF = df.select_dtypes(exclude=numerics)
numDF = df.select_dtypes(include=numerics)


# In[63]:


print(catDF.head())
numDF.head()


# In[64]:


# Scale all numeric columns
numDF = pd.DataFrame(scaler.fit_transform(numDF.values), columns=numDF.columns, index=numDF.index)
numDF.head()


# In[65]:


# Drop the target variable from the DF
catDF.drop(['Gender'], axis=1, inplace=True)


# In[66]:


# Converting all the categorical variables to dummy variables
catDF = pd.get_dummies(catDF)
catDF.shape


# In[67]:


catDF.head(2)


# In[68]:


# Preparing the Y variable
Y = df['Gender']
# Tree models have trouble turning strings to float to labeling the target variable so there is a complete feature matrix
Y = Y.replace(to_replace=['No','Yes'],value=[0,1])
print(Y.shape)


# In[69]:


# Merging with the original data frame
# Preparing the X variables
X = pd.concat([catDF, numDF], axis=1)
print(X.shape)


# In[70]:


X.head()


# In[71]:


from sklearn.model_selection import train_test_split
# Using train_test_split to Split Data into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)


# In[72]:


# First we build and train our Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=5, random_state=42, n_estimators = 300).fit(X_train, y_train)
rf.feature_importances_
# create a new DataFrame with feature importances and column names
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})

# sort the features by importance
feature_importances = feature_importances.sort_values('importance', ascending=False)

# print the feature importances
print(feature_importances)


# In[73]:


# Permutation Importance
from sklearn.inspection import permutation_importance
r = permutation_importance(rf, X_test, y_test,
                           n_repeats=10,
                           random_state=0)
perm = pd.DataFrame(columns=['AVG_Importance', 'STD_Importance'], index=[i for i in X_train.columns])
perm['AVG_Importance'] = r.importances_mean


# In[74]:


print(perm.to_string())


# In[75]:


# Sorting the DataFrame by 'AVG_Importance' in descending order
perm_sorted = perm.sort_values(by='AVG_Importance', ascending=False)

# Printing the sorted DataFrame
print(perm_sorted.to_string())


# In[77]:


model = LogisticRegression(max_iter=1000)
model.fit(X,Y)
importance = model.coef_[0]
importance = np.sort(importance)
importance
importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
importance = importance.sort_values('importance', ascending=False)

# create a chart of feature importances
plt.figure(figsize=(10,5))
plt.bar(importance['feature'], importance['importance'])
plt.xticks(rotation=90)
plt.show()


# In[78]:


X.head()


# In[79]:


type(X)


# # Bonus Challenge: Binary Classification & Resampling

# In[86]:


# Assuming 'X' and 'Y' are your DataFrames
merged_df = pd.merge(X, Y, left_index=True, right_index=True)

# If the index is not used for merging, you can specify the column names
# merged_df = pd.merge(X, Y, left_on='common_column_in_X', right_on='common_column_in_Y')

# Print the merged DataFrame
print(merged_df)


# In[87]:


merged_df.head()


# In[88]:


# if the target (class in the case of mushrooms) is text/object... then we can make the target numeric with the LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate the LabelEncoder
le = LabelEncoder()

# Fit and transform the binary column
merged_df['Gender'] = le.fit_transform(merged_df['Gender'])

# Print the mapping
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(mapping)


# In[89]:


merged_df.head()


# In[94]:


X = merged_df.drop(['Gender'], axis=1)


# In[95]:


y = merged_df['Gender']


# In[96]:


X.head()


# In[97]:


X = pd.get_dummies(X)


# In[98]:


X.head()


# In[99]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[100]:


X_test.info()


# In[105]:


from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('SGD', SGDClassifier()))
models.append(('Ridge', RidgeClassifier()))
models.append(('PAC', PassiveAggressiveClassifier()))
models.append(('Perceptron', Perceptron()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NearestCentroid', NearestCentroid()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('ExtraTree', ExtraTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('BNB', BernoulliNB()))
# models.append(('ComplementNB', ComplementNB()))
# models.append(('MultinomialNB', MultinomialNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('NuSVC', NuSVC()))
models.append(('LinearSVC', LinearSVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('ExtraTrees', ExtraTreesClassifier()))
models.append(('Bagging', BaggingClassifier()))
models.append(('AdaBoost', AdaBoostClassifier()))
models.append(('MLP', MLPClassifier()))


# In[107]:


from time import time
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

results = []
names = []
scoring = 'roc_auc'
for name, model in models:
    start = time()
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    train_time = time() - start
    start = time()  # Reset start time for prediction time measurement
    model.fit(X_train, y_train)
    predict_time = time() - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    print("Score for each of the 10 K-fold tests: ", cv_results)
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print()

plt.figure(figsize=(15, 10))
plt.boxplot(results)
plt.xticks(range(1, len(names) + 1), names, rotation=45)
plt.title('Algorithm Comparison')
plt.show()


# In[111]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import BaggingClassifier

# Fit the model
bc = BaggingClassifier().fit(X_train, y_train)

# Make predictions
y_pred = bc.predict(X_test)

# Evaluate predictions
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))


# In[112]:


merged_data.info()


# In[ ]:




