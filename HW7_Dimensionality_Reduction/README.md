# HW 7: Dimensionality reduction

The goal is to build machine learning models to estimate one's public speaking anxiety from bio-behavioral data. The data comes from the VerBio dataset, which was collected with the goal to better understand individuals' affective responses while performing public speaking tasks. More details about the dataset, including the experimental setup and type of data, can be found here: https://hubbs.engr.tamu.edu/resources/verbio-dataset/.

The data is uploaded on CANVAS and includes 55 participants. We have one presentation for each participant, which results in 55 data samples. The "data.csv" file that contains a set of bio-behavioral features (i.e., skin conductance level, skin conductance response amplitude, skin conductance response frequency, heart rate, wrist acceleration, interbeat interval, speech energy, 12 speech mel-frequency cepstral coefficients (MFCCs), speech zero crossing rate, speech voicing probability, speech fundamental frequency (F0), speech pause frequency), label (i.e., state anxiety), and participants' language information (i.e., native/non-native English speaker). Each row in the csv file corresponds to one data sample. The bio-behavioral features were computed based on the entire presentation for each participant.

## (a) Data pre-processing
Identify missing data values and replace them with the corresponding feature mean.


```python
import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
```


```python
# read csv file 
data = []
data_path = "data.csv"

with open(data_path, 'r', encoding='utf-8-sig') as f:
    csv_file = csv.reader(f)
    data = [row for row in csv_file]
```


```python
df = pd.DataFrame(data)
new_header = df.iloc[0] # grab the first row for the header
df = df[1:] # take the data less the header row
df.columns = new_header
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PID</th>
      <th>SCL</th>
      <th>SCRamp</th>
      <th>SCRfreq</th>
      <th>HR</th>
      <th>BVP</th>
      <th>TEMP</th>
      <th>ACC</th>
      <th>IBI</th>
      <th>RMSenergy</th>
      <th>...</th>
      <th>mfcc[9]</th>
      <th>mfcc[10]</th>
      <th>mfcc[11]</th>
      <th>mfcc[12]</th>
      <th>zcr</th>
      <th>voiceProb</th>
      <th>F0</th>
      <th>pause_frequency</th>
      <th>StateAnxiety</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>P001</td>
      <td>0.353694542</td>
      <td>0.049017498</td>
      <td>9.649805447</td>
      <td>79.04321244</td>
      <td>0.03971799</td>
      <td>33.16005188</td>
      <td>62.89235567</td>
      <td>0.859414333</td>
      <td>0.027971981</td>
      <td>...</td>
      <td>-2.724729485</td>
      <td>-3.590817861</td>
      <td>0.39038344</td>
      <td>-11.85513391</td>
      <td>0.071546573</td>
      <td>0.516881484</td>
      <td>95.47320932</td>
      <td>3.857142857</td>
      <td>62</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P003</td>
      <td>0.424881154</td>
      <td>0.040277009</td>
      <td>12.0754717</td>
      <td>90.67472803</td>
      <td>-0.012489676</td>
      <td>33.560587</td>
      <td>66.50715626</td>
      <td></td>
      <td>0.025150245</td>
      <td>...</td>
      <td>-0.994850967</td>
      <td>-5.456365037</td>
      <td>-1.185722752</td>
      <td>-4.423543868</td>
      <td>0.088058334</td>
      <td>0.493522682</td>
      <td>77.45466101</td>
      <td>0.888888889</td>
      <td>41</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P004</td>
      <td>0.164890091</td>
      <td>0.006638833</td>
      <td>1.730769231</td>
      <td>82.39349359</td>
      <td>0.162948012</td>
      <td>31.74230769</td>
      <td>65.21567384</td>
      <td>0.774073846</td>
      <td>0.021411449</td>
      <td>...</td>
      <td>-7.322685216</td>
      <td>-4.966898789</td>
      <td>-8.985444317</td>
      <td>-8.322640694</td>
      <td>0.07327186</td>
      <td>0.581679662</td>
      <td>132.8558223</td>
      <td>1</td>
      <td>39</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P005</td>
      <td>0.85070951</td>
      <td>0.040912997</td>
      <td>13.97727273</td>
      <td>87.31784091</td>
      <td>-0.011401703</td>
      <td>31.15701705</td>
      <td>64.43825796</td>
      <td>0.448681143</td>
      <td>0.008874291</td>
      <td>...</td>
      <td>-8.647626987</td>
      <td>1.977405603</td>
      <td>-0.627533467</td>
      <td>-1.1523338</td>
      <td>0.052280549</td>
      <td>0.51024507</td>
      <td>96.28413316</td>
      <td>2.428571429</td>
      <td>40</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>P006</td>
      <td>2.918752269</td>
      <td>0.175392157</td>
      <td>18.88594164</td>
      <td>80.05480565</td>
      <td>0.001135861</td>
      <td>33.05362511</td>
      <td>63.15713836</td>
      <td>0.734409</td>
      <td>0.01189714</td>
      <td>...</td>
      <td>6.593793786</td>
      <td>3.238551575</td>
      <td>7.198553223</td>
      <td>1.251417088</td>
      <td>0.027679581</td>
      <td>0.437491946</td>
      <td>34.32755922</td>
      <td>1.818181818</td>
      <td>50</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
# convert data types for the features 
features = df.columns[1:]
features = df[features].apply(pd.to_numeric, errors='coerce')
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SCL</th>
      <th>SCRamp</th>
      <th>SCRfreq</th>
      <th>HR</th>
      <th>BVP</th>
      <th>TEMP</th>
      <th>ACC</th>
      <th>IBI</th>
      <th>RMSenergy</th>
      <th>mfcc[1]</th>
      <th>...</th>
      <th>mfcc[9]</th>
      <th>mfcc[10]</th>
      <th>mfcc[11]</th>
      <th>mfcc[12]</th>
      <th>zcr</th>
      <th>voiceProb</th>
      <th>F0</th>
      <th>pause_frequency</th>
      <th>StateAnxiety</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.353695</td>
      <td>0.049017</td>
      <td>9.649805</td>
      <td>79.043212</td>
      <td>0.039718</td>
      <td>33.160052</td>
      <td>62.892356</td>
      <td>0.859414</td>
      <td>0.027972</td>
      <td>1.812113</td>
      <td>...</td>
      <td>-2.724729</td>
      <td>-3.590818</td>
      <td>0.390383</td>
      <td>-11.855134</td>
      <td>0.071547</td>
      <td>0.516881</td>
      <td>95.473209</td>
      <td>3.857143</td>
      <td>62.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.424881</td>
      <td>0.040277</td>
      <td>12.075472</td>
      <td>90.674728</td>
      <td>-0.012490</td>
      <td>33.560587</td>
      <td>66.507156</td>
      <td>NaN</td>
      <td>0.025150</td>
      <td>0.895732</td>
      <td>...</td>
      <td>-0.994851</td>
      <td>-5.456365</td>
      <td>-1.185723</td>
      <td>-4.423544</td>
      <td>0.088058</td>
      <td>0.493523</td>
      <td>77.454661</td>
      <td>0.888889</td>
      <td>41.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.164890</td>
      <td>0.006639</td>
      <td>1.730769</td>
      <td>82.393494</td>
      <td>0.162948</td>
      <td>31.742308</td>
      <td>65.215674</td>
      <td>0.774074</td>
      <td>0.021411</td>
      <td>-0.813331</td>
      <td>...</td>
      <td>-7.322685</td>
      <td>-4.966899</td>
      <td>-8.985444</td>
      <td>-8.322641</td>
      <td>0.073272</td>
      <td>0.581680</td>
      <td>132.855822</td>
      <td>1.000000</td>
      <td>39.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.850710</td>
      <td>0.040913</td>
      <td>13.977273</td>
      <td>87.317841</td>
      <td>-0.011402</td>
      <td>31.157017</td>
      <td>64.438258</td>
      <td>0.448681</td>
      <td>0.008874</td>
      <td>-0.676535</td>
      <td>...</td>
      <td>-8.647627</td>
      <td>1.977406</td>
      <td>-0.627533</td>
      <td>-1.152334</td>
      <td>0.052281</td>
      <td>0.510245</td>
      <td>96.284133</td>
      <td>2.428571</td>
      <td>40.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.918752</td>
      <td>0.175392</td>
      <td>18.885942</td>
      <td>80.054806</td>
      <td>0.001136</td>
      <td>33.053625</td>
      <td>63.157138</td>
      <td>0.734409</td>
      <td>0.011897</td>
      <td>-0.329363</td>
      <td>...</td>
      <td>6.593794</td>
      <td>3.238552</td>
      <td>7.198553</td>
      <td>1.251417</td>
      <td>0.027680</td>
      <td>0.437492</td>
      <td>34.327559</td>
      <td>1.818182</td>
      <td>50.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python

```

## (b) Data exploration 
Provide visualizations of some of the features with respect to the state anxiety label (e.g., overlaying histograms, scatter plots), and quantify associations between the features and the label (e.g., via correlation coefficient). Please discuss your findings.


```python
# plot the 2-D scatter plots
index_numerical = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 22, 23, 24, 26]

sns.set_theme(style='white')
for i in index_numerical:
    sns.scatterplot(data= features, x=features[features.columns[i]], y=features['StateAnxiety'])
    plt.title(f"{features.columns[i]} - {'StateAnxiety'}")
    plt.show()
```


    
![png](output_10_0.png)
    



    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    



    
![png](output_10_3.png)
    



    
![png](output_10_4.png)
    



    
![png](output_10_5.png)
    



    
![png](output_10_6.png)
    



    
![png](output_10_7.png)
    



    
![png](output_10_8.png)
    



    
![png](output_10_9.png)
    



    
![png](output_10_10.png)
    



    
![png](output_10_11.png)
    



    
![png](output_10_12.png)
    



    
![png](output_10_13.png)
    



    
![png](output_10_14.png)
    



```python
# compute the Pearson's correlation coefficients

list_corr_coef = []
index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26]

for i in index:
    list_corr_coef.append(np.corrcoef(features[features.columns[i]], features['StateAnxiety'])[0, 1])

for i, n in zip(index, range(26)):
    print(f"Pearson's correlation coefficient between {features.columns[i]} and {'StateAnxiety'}: {list_corr_coef[n]}")
```

    Pearson's correlation coefficient between SCL and StateAnxiety: -0.049804667294520925
    Pearson's correlation coefficient between SCRamp and StateAnxiety: -0.014173697358366292
    Pearson's correlation coefficient between SCRfreq and StateAnxiety: 0.013869512677839924
    Pearson's correlation coefficient between HR and StateAnxiety: -0.2170424118534046
    Pearson's correlation coefficient between BVP and StateAnxiety: -0.00035485586179540083
    Pearson's correlation coefficient between TEMP and StateAnxiety: -0.12054102158121277
    Pearson's correlation coefficient between ACC and StateAnxiety: -0.216606272115361
    Pearson's correlation coefficient between IBI and StateAnxiety: 0.18315643320166755
    Pearson's correlation coefficient between RMSenergy and StateAnxiety: -0.23609034514486693
    Pearson's correlation coefficient between mfcc[1] and StateAnxiety: -0.12590333392378003
    Pearson's correlation coefficient between mfcc[2] and StateAnxiety: 0.08125579732950128
    Pearson's correlation coefficient between mfcc[3] and StateAnxiety: -0.10599633731007585
    Pearson's correlation coefficient between mfcc[4] and StateAnxiety: 0.0966262952555937
    Pearson's correlation coefficient between mfcc[5] and StateAnxiety: 0.004174562312476519
    Pearson's correlation coefficient between mfcc[6] and StateAnxiety: -0.017387613666625643
    Pearson's correlation coefficient between mfcc[7] and StateAnxiety: -0.037643549662553885
    Pearson's correlation coefficient between mfcc[8] and StateAnxiety: 0.05847540482073137
    Pearson's correlation coefficient between mfcc[9] and StateAnxiety: -0.16773197748262772
    Pearson's correlation coefficient between mfcc[10] and StateAnxiety: 0.12559754274594837
    Pearson's correlation coefficient between mfcc[11] and StateAnxiety: -0.016609137303813536
    Pearson's correlation coefficient between mfcc[12] and StateAnxiety: 0.04537417978830816
    Pearson's correlation coefficient between zcr and StateAnxiety: 0.024935222423946832
    Pearson's correlation coefficient between voiceProb and StateAnxiety: -0.04201380149878644
    Pearson's correlation coefficient between F0 and StateAnxiety: -0.01798161988784301
    Pearson's correlation coefficient between pause_frequency and StateAnxiety: 0.07745974491905257
    Pearson's correlation coefficient between Language and StateAnxiety: -0.06544385767093154
    

**IBI** (0.18) and **mfcc_10** (0.126) have the highest Pearson's correlation coefficient with StateAnxiety among the others. (**positive**)

**RMSenergy** (-0.236) and **HR** (-0.217) have the lowest Pearson's correlation coefficient with StateAnxiety among the others. (**negative**)

## (c) Data exploration
Provide visualizations of some of the features with respect to the participants' language information (e.g., bar plots for Native and Non-Native speakers), and identify whether these features depict differences between Native and Non-Native English
speakers (1: native, 2: non-native in "data.csv") (e.g., via Fisher's criterion). What might be potential sources of these differences (if any)?


```python
# create a figure with 9 subplots
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (15, 10))

sns.barplot(x = "Language", y = "SCL", data = features, ax = axes[0,0])
sns.barplot(x = "Language", y = "SCRamp", data = features, ax = axes[0,1])
sns.barplot(x = "Language", y = "SCRfreq", data = features, ax = axes[0,2])
sns.barplot(x = "Language", y = "HR", data = features, ax = axes[1,0])
sns.barplot(x = "Language", y = "BVP", data = features, ax = axes[1,1])
sns.barplot(x = "Language", y = "TEMP", data = features, ax = axes[1,2])
sns.barplot(x = "Language", y = "pause_frequency", data = features, ax = axes[2,0])
sns.barplot(x = "Language", y = "StateAnxiety", data = features, ax = axes[2,1])
sns.barplot(x = "Language", y = "RMSenergy", data = features, ax = axes[2,2])

# adjust the layout
plt.tight_layout()

# show the plot
plt.show()
```


    
![png](output_14_0.png)
    


**Conclusions**: The mean of SCL, BVP and RMSenergy are higher for non-Native speakers than Native speakers. The mean of SCRamp, SCRfreq and pause_frequency are higher for Native speakers than non-Native speakers. The potential sources of these differences might be attributed to the stimuli, such as emotional or physiological arousal being perceived by the speakers. It is anticipated that native speakers would be less nervous and more comfortable in presentation comparing to non-native speakers. However, the mean of skin conductance response amplitude (SCRamp), skin conductance response frequency (SCRfreq) and speech pause frequency are higher for native speaker which is counterintuitive and suggests that languages might not be the most important thing for evaluating one's public speaking anxiety. Other factors such as personality or experiences might impact the results and can be further explored. 

## (d) Feature selection for state anxiety
Explore a filter feature selection method of your choice to identify the bio-behavioral features that are the most informative
of the state anxiety label. Using any machine learning algorithm that you would like, plot the absolute error between the actual and predicted state anxiety values using a 5-fold cross-validation (i.e., average over the 5 folds) against different number of selected features. Please discuss the results.


```python
def rf_regression(X, y, num_features):
    # Initialize the random forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Initialize the KFold object for cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize the list to store the absolute errors for each fold
    abs_errors = []

    # Perform cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :].iloc[:, :num_features], X.iloc[test_index, :].iloc[:, :num_features]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the random forest regressor on the training data
        rf.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = rf.predict(X_test)

        # Calculate the absolute error between the actual and predicted values
        abs_error = mean_absolute_error(y_test, y_pred)

        # Add the absolute error to the list
        abs_errors.append(abs_error)

    # Calculate the average absolute error across the folds
    avg_abs_error = sum(abs_errors) / n_folds

    return avg_abs_error
```

Use Correlation as the filter feature selection method

Model 1: IBI & mfcc_10 as input features


```python
X = features.iloc[:, [7,18]] # IBI & mfcc_10 as input features 
y = features[["StateAnxiety"]]

# Define the number of folds for cross-validation
n_folds = 5 

# Initialize the list to store the absolute errors for each model 
abs_errors = []
```


```python
abs_errors.append(rf_regression(X,y,2))
abs_errors
```

    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    




    [7.640619528619529]



Model 2: IBI & mfcc_10 & mfcc_4 as input features


```python
X = features.iloc[:, [7,12,18]] # IBI & mfcc_10 & mfcc_4 as input features 
y = features[["StateAnxiety"]]
```


```python
abs_errors.append(rf_regression(X,y,2))
abs_errors
```

    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    




    [7.640619528619529, 8.426383838383838]



Model 3: RMSenergy & HR as input features 


```python
X = features.iloc[:, [3,8]] # HR & RMSenergy as input features 
y = features[["StateAnxiety"]]
```


```python
abs_errors.append(rf_regression(X,y,2))
abs_errors
```

    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    




    [7.640619528619529, 8.426383838383838, 8.387892255892258]



Model 4: IBI & mfcc_10 & RMSenergy & HR as input features


```python
X = features.iloc[:, [3,7,8,18]] # HR & RMSenergy & IBI & mfcc_10 as input features 
y = features[["StateAnxiety"]]
```


```python
abs_errors.append(rf_regression(X,y,4))
abs_errors
```

    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    




    [7.640619528619529, 8.426383838383838, 8.387892255892258, 7.649562289562292]




```python
# Plot the results using a line graph

models = ['1','2','3','4']

plt.plot(models, abs_errors, 'bo-')
plt.xlabel('Models')
plt.ylabel('Average absolute error')
plt.title('Random Forest Regression Performance')
plt.show()
```


    
![png](output_31_0.png)
    


**Conclusion**: **Model 1** with 2 variables having the highest Pearson's correlation coefficient with StateAnxiety has the **lowest average absolute error**. **Model 2** combining the variables of Model 1 plus a variable with third highest correlation coefficient has the **highest average absolute error**. The error of Model 3 with 2 variables having the lowest Pearson's correlation coefficient with StateAnxiety dropped a bit comparing to Model 2, but was the second highest. Model 4 with 4 variables having 2 highest and 2 lowest correlation coefficient respectively performed almost the same with Model 1. Therefore, from the results we can conclude that model 1 including 2 variables with the highest Pearson's correlation coefficient predicts the state anxiety label most accurately.

Use different number of selected features in the model.


```python
indexes = range(0,25)
mean_error = []

for i in indexes: 
    X = features.iloc[:, :i+1]
    y = features[["StateAnxiety"]]
    
    mean_error.append(rf_regression(X, y, i+1))
mean_error
```

    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3807870786.py:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
    




    [10.339589225589226,
     9.642060606060605,
     9.742404040404042,
     8.971696969696971,
     8.298632996632998,
     8.489158249158248,
     8.167151515151517,
     8.017063973063973,
     7.800350168350168,
     7.855010101010102,
     7.782653198653199,
     8.013542087542088,
     7.846255892255894,
     7.9834006734006735,
     7.622686868686867,
     7.968404040404041,
     7.893084175084175,
     7.911239057239058,
     7.536781144781145,
     7.8677777777777775,
     7.7831515151515145,
     7.780127946127948,
     7.476289562289563,
     7.55270707070707,
     7.531797979797981]




```python
# Plot the results using a line graph

num_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20, 21, 22, 23, 24]

plt.plot(num_features, mean_error, 'bo-')
plt.xlabel('Number of selected features')
plt.ylabel('Average absolute error')
plt.title('Random Forest Regression Performance')
plt.show()
```


    
![png](output_35_0.png)
    


**Conclusion**: From the graph, we can see that the performance of the random forest regression improves as the number of selected features increases up to around 8, and then starts to level off.

## (e) Feature selection for Native English speaker classification
Explore a filter feature selection method of your choice to identify the bio-behavioral features that are the most informative of whether a speaker is a Native English speaker or not (1: native, 2: non-native in data.csv"). Using any machine learning algorithm that you would like, plot the binary classification accuracy between Native and Non-Native English speakers using a 5-fold cross-validation (i.e., average over the 5 folds) against different number of selected features. Please discuss the results.

Use Correlation as the filter feature selection method


```python
# compute the Pearson's correlation coefficients

list_corr_coef = []
index = range(0,26)

for i in index:
    list_corr_coef.append(np.corrcoef(features[features.columns[i]], features['Language'])[0, 1])

for i in index:
    print(f"Pearson's correlation coefficient between {features.columns[i]} and {'Language'}: {list_corr_coef[i]}")
```

    Pearson's correlation coefficient between SCL and Language: 0.09008433712012787
    Pearson's correlation coefficient between SCRamp and Language: -0.13345922242883587
    Pearson's correlation coefficient between SCRfreq and Language: -0.10271543690922964
    Pearson's correlation coefficient between HR and Language: -0.17947740800825385
    Pearson's correlation coefficient between BVP and Language: 0.08679254286602066
    Pearson's correlation coefficient between TEMP and Language: -0.2433421871266833
    Pearson's correlation coefficient between ACC and Language: -0.048482295624639356
    Pearson's correlation coefficient between IBI and Language: 0.09353252697652781
    Pearson's correlation coefficient between RMSenergy and Language: 0.26436213831988736
    Pearson's correlation coefficient between mfcc[1] and Language: 0.2785420187640315
    Pearson's correlation coefficient between mfcc[2] and Language: -0.07632751027810555
    Pearson's correlation coefficient between mfcc[3] and Language: 0.13917794578134096
    Pearson's correlation coefficient between mfcc[4] and Language: -0.1296047318859937
    Pearson's correlation coefficient between mfcc[5] and Language: -0.005991563970628764
    Pearson's correlation coefficient between mfcc[6] and Language: -0.02208736013170426
    Pearson's correlation coefficient between mfcc[7] and Language: -0.018694177465410345
    Pearson's correlation coefficient between mfcc[8] and Language: -0.060680874474553885
    Pearson's correlation coefficient between mfcc[9] and Language: 0.0793716784830745
    Pearson's correlation coefficient between mfcc[10] and Language: -0.04298799096404298
    Pearson's correlation coefficient between mfcc[11] and Language: 0.17283768069734715
    Pearson's correlation coefficient between mfcc[12] and Language: -0.2728679597660959
    Pearson's correlation coefficient between zcr and Language: -0.14352880653987551
    Pearson's correlation coefficient between voiceProb and Language: 0.1953019587336696
    Pearson's correlation coefficient between F0 and Language: 0.08325108546666111
    Pearson's correlation coefficient between pause_frequency and Language: -0.2810152038368369
    Pearson's correlation coefficient between StateAnxiety and Language: -0.06544385767093153
    

**mfcc_1** (0.28) and **RMSenergy** (0.26) have the highest Pearson's correlation coefficient with Language among the others. (**positive**)

**pause_frequency** (-0.28) and **mfcc_12** (-0.27) have the lowest Pearson's correlation coefficient with Language among the others. (**negative**)

Model 1: mfcc_1 & RMSenergy as input features


```python
# Load the dataset
X = features.iloc[:, [8,9]] # mfcc_1 and RMSenergy as input features 
y = features[["Language"]]


# Define an empty list to store the classification accuracy scores
acc_scores = []
```


```python
# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
# Use 5-fold cross-validation to evaluate the classification accuracy
kf = KFold(n_splits = 5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=kf)
    
# Calculate the average classification accuracy across the folds
acc_score = sum(scores) / len(scores)
acc_scores.append(acc_score) 
```

    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    

Model 2: mfcc_1 & RMSenergy & voiceProb as input features


```python
# Load the dataset
X = features.iloc[:, [8,9,22]] # mfcc_1 and RMSenergy & voiceProb as input features 
y = features[["Language"]] 
```


```python
# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
# Use 5-fold cross-validation to evaluate the classification accuracy
kf = KFold(n_splits = 5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=kf)
    
# Calculate the average classification accuracy across the folds
acc_score = sum(scores) / len(scores)
acc_scores.append(acc_score) 
```

    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    

Model 3: pause_frequency & mfcc_12 as input features


```python
# Load the dataset
X = features.iloc[:, [20,24]] # pause_frequency & mfcc_12 as input features 
y = features[["Language"]] 
```


```python
# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
# Use 5-fold cross-validation to evaluate the classification accuracy
kf = KFold(n_splits = 5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=kf)
    
# Calculate the average classification accuracy across the folds
acc_score = sum(scores) / len(scores)
acc_scores.append(acc_score) 
```

    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    

Model 4: mfcc_1 & RMSenergy & pause_frequency & mfcc_12 as input features


```python
# Load the dataset
X = features.iloc[:, [8,9,20,24]] # mfcc_1 & RMSenergy & pause_frequency & mfcc_12 as input features 
y = features[["Language"]
```


```python
# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
# Use 5-fold cross-validation to evaluate the classification accuracy
kf = KFold(n_splits = 5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=kf)
    
# Calculate the average classification accuracy across the folds
acc_score = sum(scores) / len(scores)
acc_scores.append(acc_score) 
```

    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    


```python
# Plot the classification accuracy against the number of selected features
plt.plot(models, acc_scores, 'bo-')
plt.xlabel('Model')
plt.ylabel('Classification accuracy')
plt.title('Random Forest Classification Performance')
plt.show()
```


    
![png](output_53_0.png)
    


**Conclusions**: Model 1 with 2 variables having the highest Pearson's correlation coefficient with Language has the **lowest classification accuracy**. Adding one variable which scores the third highest for the correlation coefficient to model 1 significantly increases the accuracy. Model 3 including 2 variables having the lowest Pearson's correlation coefficient did not perform as well as Model 2 but is slightly better than Model 1 and 4. Model 4 using the above 4 features did not perform very well whose accuracy is only slightly higher than Model 1. 

Use different number of selected features in the model.


```python
indexes = range(0,26)
accuracy = []

for i in indexes: 
    X = features.iloc[:, :i+1]    
    y = features[["Language"]] 
    
    # Create a random forest classifier with 100 trees
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Use 5-fold cross-validation to evaluate the classification accuracy
    kf = KFold(n_splits = 5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=kf)
    
    # Calculate the average classification accuracy across the folds
    acc_score = sum(scores) / len(scores)
    accuracy.append(acc_score) 
        
accuracy
```

    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    




    [0.6181818181818182,
     0.7090909090909092,
     0.7454545454545454,
     0.7636363636363636,
     0.6545454545454545,
     0.7454545454545455,
     0.7090909090909092,
     0.7454545454545455,
     0.6545454545454545,
     0.7090909090909091,
     0.6545454545454545,
     0.6727272727272727,
     0.6545454545454545,
     0.6545454545454545,
     0.6727272727272727,
     0.6727272727272726,
     0.6727272727272727,
     0.7090909090909091,
     0.6727272727272726,
     0.6909090909090909,
     0.7090909090909091,
     0.6909090909090909,
     0.7272727272727273,
     0.7090909090909091,
     0.6909090909090909,
     0.7090909090909091]




```python
# Plot the results using a line graph

#num_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20, 21, 22, 23, 24]

plt.plot(range(1,27), accuracy, 'bo-')
plt.xlabel('Number of selected features')
plt.ylabel('Classification accuracy')
plt.title('Random Forest Classification Performance')
plt.show()

```


    
![png](output_57_0.png)
    


**Conclusion**: From the graph, we can see that model with 4 features has the highest accuracy. The accuracy of adding the fifth feature dropped a lot while the accuracy of adding more features oscillates up and down. 

## (f) Removing individual differences
Remove the most informative features of the Native/Non-Native English speakers from the original feature set. Using the revised
feature set, use the same machine learning algorithm as in (d) to estimate state anxiety. Please report the results similar to (d) and discuss your findings.

Remove mfcc_1 & RMSenergy & pause_frequency & mfcc_12 features from the original feature set


```python
# Load the dataset
X = features.iloc[:, [0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,18,19,21,22,23,25]] 
y = features[["Language"]]
```


```python
# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
# Use 5-fold cross-validation to evaluate the classification accuracy
kf = KFold(n_splits = 5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=kf)
    
# Calculate the average classification accuracy across the folds
acc_score = sum(scores) / len(scores)
```

    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    


```python
print(f"The Classification Accuracy is {acc_score}") 
```

    The Classification Accuracy is 0.6909090909090909
    

**Conclusion**: Comparing to the best model in (d), the accuracy of (f) is not as high as the highest accuracy (0.76) but performs average in the different number of selected features graph and has similar result with Model 2. 

## (g) Feature transformation
Split the original data samples randomly into training (80%) and testing (20%). Use principal component analysis (PCA) to reduce the dimensionality of the original features. Estimate the PCA eigenvalues and eigenvectors based on the training data. Use the PCA components learned based on the training data to transform the testing data. Provide a plot of the eigenspectrum (i.e., sum of eigenvalues with increasing number of principal components) for the training data. Based on the eigenspectrum, choose an optimal number M* of principal components. Based on this optimal number M*, train any machine learning that you would like on state anxiety and report the absolute error between the actual and predicted state anxiety values on the testing data.


```python
X = features.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26]] 
y = features[["StateAnxiety"]]
```


```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit PCA to training data
pca = PCA()
pca.fit(X_train)

# Plot eigenspectrum
eigenvalues = pca.explained_variance_
plt.plot(np.cumsum(eigenvalues))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Choose optimal number of components based on eigenspectrum
M_star = np.where(np.cumsum(eigenvalues)/np.sum(eigenvalues) >= 0.95)[0][0] + 1
print(f'Optimal number of components: {M_star}')

# Transform training and testing data using PCA
X_train_pca = pca.transform(X_train)[:, :M_star]
X_test_pca = pca.transform(X_test)[:, :M_star]

# Train random forest model on training data
rf = RandomForestRegressor()
rf.fit(X_train_pca, y_train)

# Make predictions on testing data
y_pred = rf.predict(X_test_pca)

# Calculate mean absolute error on testing data
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean absolute error: {mae}')


```


    
![png](output_67_0.png)
    


    Optimal number of components: 4
    Mean absolute error: 10.014781144781146
    

    C:\Users\MEI-KUEI LU\AppData\Local\Temp\ipykernel_2104\3258384678.py:25: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train_pca, y_train)
    


```python
# Print eigenvectors
eigenvectors = pca.components_
print(f'Eigenvectors:\n {eigenvectors}')
```

    Eigenvectors:
     [[-8.08428243e-03 -1.94105519e-04  5.87422246e-03  3.39908491e-02
       3.21055129e-04  5.17013668e-03  2.01321327e-03 -3.92266269e-04
      -1.61035811e-05  4.82787395e-03 -4.19972480e-03 -2.30227658e-02
       8.04920479e-03 -5.35957262e-02  2.24116365e-02  4.07922566e-03
      -4.00910706e-02 -2.11247512e-02 -2.14421811e-02 -2.23121870e-02
      -6.75365026e-03 -2.85792887e-04  1.69381165e-03  9.95773306e-01
      -7.89069578e-03 -1.50665900e-04]
     [-8.15728190e-03 -1.23575957e-03 -2.39292079e-01 -9.43883742e-01
       4.14116404e-04 -2.11022798e-02 -1.28347521e-02  1.08582184e-03
       1.13796113e-04 -6.13268103e-04 -9.20379095e-02  4.67926194e-02
      -1.00244376e-01 -2.28385401e-02 -7.20828407e-03 -4.12178955e-02
       2.33085700e-02 -1.00566194e-01 -2.42504716e-02 -5.64843920e-03
      -1.25681370e-01  5.31938218e-04  6.21802992e-04  3.16212984e-02
       9.59250070e-04  9.31992419e-03]
     [ 1.08272839e-02  1.56759471e-03  2.78571717e-02 -5.17727605e-04
       7.69386442e-05 -1.84705188e-02 -7.48309374e-03  2.13799401e-03
       1.38098381e-04  1.09042685e-01  4.47738804e-02  1.34774514e-01
       2.05164802e-01 -5.74485633e-01  3.97498254e-01 -4.03506607e-01
      -4.25903754e-01  1.74588016e-01 -1.53132641e-01  1.34083183e-02
      -1.81322923e-01 -2.79725865e-04  9.50713579e-05 -5.43322008e-02
       4.79812545e-02  7.59335809e-03]
     [ 7.74926732e-02  3.77808004e-03  3.41678408e-01 -2.18643555e-01
      -2.59299907e-03 -7.03982972e-03 -4.19191139e-02 -1.90353659e-03
      -2.24076581e-04  1.01605593e-01  2.52792104e-01  2.28744641e-01
       3.46411360e-01  1.77092168e-01  2.30339467e-01  2.87587223e-01
       9.21275162e-02  4.32005043e-01  2.31563719e-01  3.79246788e-01
       9.68366965e-02 -3.34656045e-03 -2.28815097e-04  4.02478340e-02
       1.01137939e-01  1.15938000e-02]
     [-1.20943671e-01 -8.97561850e-03 -6.85156500e-01  1.94351298e-01
       1.81859418e-03 -5.70752247e-02  3.21904182e-02  1.89502830e-03
       6.16883637e-04  1.36794294e-01 -9.23863466e-02  2.93664657e-01
      -2.30337004e-01  1.07102545e-01  3.50240640e-02  1.69246127e-01
      -8.56547223e-02  3.35333377e-01 -7.21264150e-02  3.44043632e-01
      -1.37587517e-01 -1.88995778e-03  6.63909377e-04  1.70269752e-02
      -5.05619162e-02  2.75870369e-02]
     [-6.64843376e-02 -5.65474111e-03 -1.83465975e-01 -2.99789316e-02
      -3.78528908e-04  3.30946363e-02  1.79485621e-01  3.08089422e-03
      -8.23356015e-04 -2.34166308e-01  2.99722516e-01 -1.29943734e-01
      -1.36484380e-01  2.81070735e-01  3.34605189e-01 -5.44227657e-01
       2.37226687e-01  2.41195829e-01 -2.64001745e-02 -8.10120608e-02
       3.70582592e-01  1.44495440e-05 -1.36600874e-03  2.59776488e-02
      -2.23315523e-02 -1.11377273e-02]
     [ 7.84904871e-02  3.72020161e-03  4.70190555e-01  6.29552764e-04
       2.15803222e-03  2.19607261e-02  1.38041075e-01 -2.46062949e-03
       1.04443070e-03  7.88673805e-02 -3.41157141e-01  4.50551330e-01
      -4.95572844e-01  2.35446691e-02  3.13699840e-03 -1.95833559e-01
       2.32849344e-01  1.84520925e-01 -1.37676587e-01 -1.96436837e-02
      -1.20528416e-01  1.70226965e-03  2.35999260e-03  2.00328354e-02
      -1.36997475e-01  4.20899222e-02]
     [ 1.42492261e-01  8.61433467e-03 -1.56080112e-01  6.22953275e-02
       9.52605682e-03  4.78702907e-02 -5.16559546e-02  1.18006277e-04
      -3.65331755e-04  1.59893130e-01  2.63609147e-01  5.71078330e-01
      -7.63661417e-02  4.71816371e-02  1.55000274e-01  4.83916340e-03
      -5.18859520e-02 -3.35380419e-01  4.55523476e-01 -3.96140983e-01
       6.75819130e-02 -2.75863154e-03 -8.00871416e-04  5.91349219e-03
       9.26297041e-02 -1.56365436e-02]
     [-4.12929817e-02  3.19578004e-03 -9.21949857e-02  8.65611978e-02
      -1.98025978e-04 -1.13269062e-01  1.12126446e-01 -3.17074600e-03
      -5.30652147e-04  1.50458348e-01 -9.16426642e-02  7.89822351e-03
       2.43574615e-01 -1.78717354e-01 -6.13884079e-02 -3.63135517e-01
       4.94959851e-01 -3.15354525e-01  3.31769925e-01  4.47886263e-01
      -1.67495387e-01 -1.84517835e-03  8.79264002e-04  1.63390763e-02
      -1.14843732e-01  4.74802783e-02]
     [-3.65915438e-03  1.06857020e-02 -1.03464833e-01  1.17336211e-02
      -1.86414810e-03 -1.66195745e-02 -1.07316691e-01  2.74859621e-03
      -2.77467896e-04 -3.50834492e-01 -3.29261463e-01  6.33295781e-02
      -6.45219186e-02 -4.93278671e-01  3.05622698e-01  3.24459095e-01
       2.89764710e-01 -1.19525441e-02  6.64429812e-02 -6.20660668e-04
       4.49907933e-01  1.82455965e-03 -1.57011176e-03 -1.57147667e-02
       4.79948461e-02 -4.53456704e-02]
     [-1.59847793e-01 -1.42872819e-02 -8.59527636e-03 -6.58884059e-02
       2.09253666e-03 -1.14246971e-01  2.09818042e-01  6.30176219e-04
       3.10095820e-04  5.20999292e-01 -4.67238077e-03 -2.71200484e-01
      -7.91590218e-02 -2.25367082e-01 -7.25552149e-02  1.07145899e-01
       1.60510282e-02  3.08716657e-01  3.14841882e-01 -2.94599851e-01
       1.95658353e-01 -3.65813838e-03  5.24044300e-03 -1.25543851e-02
      -4.16867317e-01  1.40950557e-02]
     [-5.36246773e-02 -5.34292997e-03  3.96618779e-02 -4.22208061e-02
       6.87446555e-03 -4.23305619e-01 -2.14605804e-02 -1.56006980e-03
      -3.52766420e-04  1.59451029e-01 -8.25350830e-02  2.72072929e-01
       2.01041294e-01  1.46938118e-01 -1.00319712e-01 -8.47242818e-02
      -2.22968343e-01 -2.61299100e-01 -3.34898784e-01  1.46411181e-01
       5.52569089e-01 -1.32795080e-03  1.38903450e-03  2.69550324e-04
      -2.64673279e-01  3.94628469e-02]
     [ 4.45055939e-01  2.41622613e-02 -6.53898000e-02  4.67894384e-02
       1.67802803e-05 -2.66791299e-01 -4.06440099e-01 -2.18311537e-03
      -4.10267334e-04  4.20945285e-01 -2.22138858e-01 -2.82398010e-01
      -7.36888607e-02  1.92174857e-01  3.53984926e-01 -3.91090818e-02
       1.57630445e-01 -1.18398766e-02 -9.33809303e-02 -7.23740779e-02
      -3.35428602e-02 -1.76251225e-03  2.07000479e-03  2.09773489e-03
       2.04941153e-01  3.05218382e-02]
     [-1.79955551e-01 -1.09452350e-02 -5.59298109e-02 -8.93307595e-04
      -2.20464935e-03 -2.14983755e-01 -6.76539952e-03  4.66657717e-03
       1.15050576e-03 -4.47432781e-02 -4.08977894e-01  1.12845644e-01
       2.66954209e-01  8.24079238e-02 -3.36184963e-01 -2.41504790e-01
      -2.59687224e-02  3.16591196e-01  2.19903297e-01 -2.45299546e-01
       8.91438114e-02  2.48973572e-03 -4.33357611e-03  2.15534508e-02
       5.14463079e-01 -6.42953357e-02]
     [-3.26851732e-01 -2.14082423e-02  5.67250514e-02  3.55329229e-02
      -6.01000349e-03 -3.81669076e-01  1.10323888e-01 -3.64750762e-03
      -3.90500914e-04 -2.63062818e-01 -2.08266508e-01 -2.32671415e-03
       1.89690145e-01  2.79010297e-01  4.46238716e-01  1.42464779e-01
      -2.28987610e-02 -7.38976784e-02  9.25477763e-02 -2.28248429e-01
      -3.76686170e-01  8.97431253e-05 -1.78541195e-03 -9.56875571e-03
      -2.73228462e-01  3.48097181e-02]
     [-4.64329810e-02 -1.75820093e-03 -4.30408110e-02  1.49369678e-03
      -1.56406485e-02  2.79799916e-01  6.48261049e-01 -1.73118888e-03
       1.20090943e-04  3.21108397e-01 -1.56995612e-01 -2.80846218e-03
       1.90953244e-01  8.13758740e-02  2.26489803e-01  1.72836535e-01
       1.04119326e-01 -1.86489621e-01 -2.87900593e-01 -6.58729026e-02
       5.51301050e-02 -9.66106391e-04  9.18120583e-04 -1.27672776e-02
       3.15697344e-01 -9.73273421e-02]
     [-6.05280277e-01 -3.54526476e-02  9.30254160e-02  4.58294427e-03
      -1.07191801e-02 -2.08035651e-01 -2.47965714e-01 -2.54133143e-02
       9.39221540e-05  2.09256407e-01  3.64311018e-01  5.33958370e-02
      -1.53645561e-01 -1.55070268e-01  4.12605112e-03  8.05376666e-02
       3.10271723e-01 -6.12608363e-02 -2.73121016e-01 -7.60210280e-02
      -6.55820307e-02 -1.99916977e-03  1.34477950e-03 -4.23436146e-03
       3.04733970e-01  8.41152721e-02]
     [-4.62683970e-02  5.39374245e-03 -1.36424170e-01  1.03421398e-02
      -2.02244388e-02  4.38312725e-01 -3.62108222e-01  1.55486651e-02
       8.68739267e-04  7.72085483e-02 -8.94922187e-02  1.89339159e-01
       4.39274405e-01  5.90716156e-02 -5.66179878e-02 -5.26989668e-02
       3.07847586e-01  1.42652851e-01 -2.75649116e-01 -2.82939743e-01
      -5.57422165e-02  2.89636745e-03  5.29999430e-03  3.27682757e-03
      -3.48296809e-01 -8.56955198e-02]
     [ 4.53260812e-01  2.75619534e-02 -1.21928780e-01  1.48093023e-02
       2.93417784e-02 -4.38000648e-01  2.87776813e-01 -1.17133688e-02
       8.58564774e-04 -1.66205456e-01  2.97029936e-01  1.07862682e-01
       1.50266041e-01 -1.95964342e-01 -2.15721810e-01  1.19250534e-01
       2.83442754e-01  1.64522061e-01 -2.54070425e-01 -2.28307933e-01
      -1.68447434e-01  2.55357408e-03 -5.57368334e-04  5.74505154e-03
      -2.12334140e-02  2.96204469e-02]
     [ 2.40390419e-02 -4.04358914e-02 -3.26887453e-02  3.29021022e-03
       3.63499198e-03  1.28816131e-01  3.60417282e-02  9.01808986e-02
       1.21963261e-03 -2.75128441e-02 -5.99532640e-02 -3.87572909e-03
       7.68620661e-02  6.96452165e-03 -3.44183888e-03  1.49984349e-02
      -8.16925612e-03  1.26218378e-02  1.40381033e-03 -6.70704037e-02
       5.05644192e-02 -9.72265030e-04  8.00927773e-03 -1.58303790e-03
       4.45662731e-02  9.75075055e-01]
     [-2.86031333e-02  1.25343967e-01  1.06167816e-02  3.48299606e-05
       5.13918956e-01 -1.36757651e-02 -3.32573466e-03  8.44119392e-01
       1.37881864e-02  1.08259181e-02  1.05159959e-02 -3.32790285e-03
      -8.36413912e-03  3.64397846e-04  6.13201661e-03  4.53401552e-03
       9.53614383e-03 -5.95434358e-03 -8.31753867e-03  7.09243669e-03
      -9.63656048e-03  2.57802075e-02 -9.81916489e-03  5.91012655e-05
       6.89966754e-03 -6.93187982e-02]
     [ 1.54023607e-02  3.40933390e-03  2.19876951e-03  8.40963196e-04
      -8.53580414e-01 -3.54696851e-02  7.05536323e-03  5.15239563e-01
       8.40757744e-03 -2.36800742e-03  2.01450087e-02  5.39692505e-03
      -1.55408516e-02 -8.41713911e-03 -9.39314340e-03  1.84531767e-03
       3.03501076e-04 -2.91467963e-03  1.65222780e-03  2.97423476e-03
      -4.35007370e-03 -6.02863217e-03 -4.48194887e-02  9.19189172e-04
      -3.44857247e-04 -3.69590073e-02]
     [ 5.47791423e-02 -9.81297121e-01  8.57006070e-04  1.35080340e-03
       6.62286413e-02 -1.06601026e-03 -5.54980224e-03  1.00243099e-01
       9.89093337e-03 -1.66941243e-03  3.38798574e-04  2.63266826e-03
       9.94850773e-04 -7.94182616e-03  2.15504717e-03  1.16425289e-03
       9.78992530e-03 -4.25880505e-03 -2.32035468e-03  4.54032383e-03
      -1.55585769e-03  3.54162418e-02 -1.24698290e-01  3.67475278e-04
      -1.42291247e-03 -4.95937478e-02]
     [-7.14387739e-03  1.15321103e-01 -5.88188887e-04 -5.40628462e-04
       6.87852029e-03  3.42560313e-03 -2.68437131e-05 -4.80624877e-02
      -1.36261132e-03  8.74927502e-03  6.16560155e-04 -7.05889434e-04
       4.04815110e-04  9.93056356e-04  1.39982221e-03  9.87664555e-04
      -2.23977242e-04  7.95602374e-04  2.66087279e-04 -2.67974718e-04
       1.44330501e-03  6.16016600e-01 -7.77452942e-01  1.39062095e-03
      -4.34696826e-03  1.62382225e-02]
     [-2.36628719e-03  5.02196914e-02  3.75654868e-04 -6.05191625e-04
       3.17718113e-02  4.05492072e-03 -1.01051497e-03 -8.94258786e-03
      -2.70633723e-02 -1.58469074e-04 -3.14472956e-03 -4.38780444e-04
       8.37167440e-04 -6.25058926e-04 -1.95563108e-03 -6.53108487e-04
       1.79317521e-03  1.26779223e-03 -4.26426900e-03 -2.90205419e-03
      -6.42385099e-04 -7.86158281e-01 -6.14403175e-01  6.95263675e-04
      -2.83951510e-03  6.37391686e-03]
     [-3.90435314e-04  9.51727526e-03 -8.22903362e-05 -3.89815572e-05
       2.96197224e-04  3.06691402e-04 -2.17875007e-04 -1.73964414e-02
       9.99449595e-01 -3.49097219e-04  4.26400895e-04 -8.63777529e-04
       1.70339601e-04  5.03989679e-04  1.18449707e-03 -2.78864090e-04
      -3.18097939e-04 -1.06513264e-03  6.57143400e-04  5.16510223e-04
       8.09070203e-04 -2.11123621e-02 -1.59648775e-02  6.32568066e-06
      -3.81922511e-04  8.62592844e-04]]
    


```python
# Print eigenvalues
eigenvalues = pca.explained_variance_
print(f'Eigenvalues:\n {eigenvalues}')
```

    Eigenvalues:
     [1.89396505e+03 9.55893443e+01 6.86526053e+01 4.30961541e+01
     1.69474712e+01 1.32156127e+01 1.03971505e+01 6.80383679e+00
     5.88289287e+00 5.50181666e+00 3.96477368e+00 2.61937221e+00
     1.97008823e+00 1.73263362e+00 1.56988725e+00 1.37842599e+00
     7.72400528e-01 6.96138075e-01 4.20881340e-01 1.10318971e-01
     6.25372478e-03 2.70448778e-03 9.11052857e-04 6.99609165e-05
     2.86873647e-05 6.10282726e-06]
    
