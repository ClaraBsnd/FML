As part of my Machine Learning class, I had the opportunity to work on two projects with my teammates, Yuhsin Liao, Lucrezia Certo, and Leopold Granger. The projects focused on Email Classification and Exoplanet Identification, and involved implementing machine learning techniques to solve real-world problems. 

# Kaggle Competition - Email Classification 
Group name on Kaggle: MLwithmeTonight (Credits to my talented teammate Yuhsin Liao)

Leaderboard: 5/25

## Context
The Kaggle competition focuses on developing a multi-class classifier to categorize emails into four classes based on their metadata. The goal is to solve the problem of finding meaningful emails among promotional emails. Further information about the task can be found in the link below: 

https://www.kaggle.com/competitions/dsba-fm-centralesupelec-ml-course

## Data
### Datasets

- train.csv - the training set
- test.csv - the test set
- sample_submission.csv - a sample submission file in the correct format
- skeleton_code.py - a skeleton python script to read train and test data and write output to the submission file.


### Data fields
<details>
  <summary> On data fields </summary>
  
  - Id - the id of the email
  - date - date and time at which mail was received
  - org - the organisation of the sender
  - tld - top-level domain of the sender's organisation
  - ccs - number of people cc'd in the email
  - bcced - is the receiver bcc'd in the email
  - mail_type - the type of the mail body
  - images - number of images in the mail body
  - urls - number of urls in the mail body
  - salutations - is salutation used in the email?
  - designation - is designation of the sender mentioned in the email?
  - chars_in_subject - number of characters in the email's subject
  - chars_in_body - number of characters in the email's body
  - label - the label of the email
  
</details>

## Code
The code for the email classification can be found in the email_classification.py file. 

As part of our data preprocessing and feature engineering, we created additional variables (such as data decomposition (`year`, `month`, etc.), insitution type, or label number), we encoded categorical variables (`mail_type`, `tld` and `org`) and numerical variables (based on the quantile we assigned the data points to). We also combined features, such as `character_count` and `image_count` as there are important characteristic of an email. Finally, we used **XGBoost** to select relevant features from the large number of variables after feature engineering, retaining only those with importance greater than 0.001 (around 10% of total features). 

Then, we tested several models, among them KNN was set as our baseline model and XGBoost was the best performing one. 

More information about our methodology and result can be found below. 

<details>

<summary> Detail on Methodology and Results </summary>

### 1. Methodology

#### 1.1. Data Exploration

After analysing the dataset, we found that:
- The date variable should be transformed into a date type so to be able to extract more specific information like the month or the day of the week the email was sent. 
- Considering that the aim is to run a classification model on the data, the categorical variables will have to go through some form of encoding.  
- We needed to investigate the distributions of the numerical variables to understand whether some transformation was needed.
- There were few NAs (4% max of the total rows). We decided not to delete them as they still hold information and instead replaced them with either 0 in numerical cases or “not defined” in categorical variables. 
- There were some rows that were exact duplicates with the exception of the label.

#### 1.2. Data Preprocessing

We transformed the `date` column into the appropriate format, datetime type.


#### 1.3. Feature Engineering

- *Date Transformation* : we decided to split the date variable into 5 new variables: `day`, `hour`, `year`, `month` and `weekday`. We did this as we believed there might be some patterns regarding the moment the email has been sent and the type of email. For example updates from the bank/insurance might be sent regularly on a specific day of the week.

- *Categorical Data Encoding* : We performed two types of **encoding** on the object variables. We performed one hot for the `mail_type` because the main information we wanted to focus on was the type associated with the email. Regarding the `org` and `tdl` variables, we performed target-encoding based on each category’s probability to target labels. For example, if the category “facebook” appears in 100 rows, 50 with label 1 and 50 with  label 2, we will create the columns “org_label_1” = 0.5 and “org_label_2” = 0.5.

- *New Variables* : We also performed an additional layer of feature engineering on the tld and org variables. By studying the values of these variables, we noticed that some of them were linked to a certain type of institution ( ex. “centralesupelec” and “iiit” are academic institutions). Because of this, we thought that having variables marking the type of institution the sender belonged to could increase the performance of the model. We then created 4 new variables -  `academic`, `government`, `e-learning` and `travel` - where we hot encoded all the observations that satisfied the belongingness criteria of each type of institution.  

- *Variable Distribution* : To solve the issue of skewness and outliers that strongly biased upward the mean of numerical data, we decided that we needed to retain whether the variable is equal to zero or not, and a general idea of the magnitude of the value. In our opinion a good way to achieve this was to encode these variables based on the quantile they belonged to. Therefore we associated each value to one of 6 bins where: 0-20% → 1, 20-40% → 2, 40-60% → 3 etc. Plus a bin 0 where all the zero values were entered (applied to all the numerical variables with the exception of “chars_in_body” as it did not have any 0 values). Hence, we generated four new variables: `quantile_chars_body`, `quantile_chars_subject` , `quantile_images`, `quantile_urls`.

- *Duplicate issue* : To handle duplicates and overlapping data between the training and test sets, we added 8 new columns initialized to 0 for labels. Using a dictionary, we checked for duplicates in the email, setting the corresponding column to 1 and keeping the current label column at 0. For the test set, we created a "label" column initialized with 9, and applied the same logic. Generated variables: `label_0`, `label_1`, `label_2`, etc.

- *Feature Combination* : We explored combining character count and image count in two ways. Firstly, by taking the maximum value between the quantile of characters and images to determine verbosity or image-heavy nature of an email. Secondly, by applying a log transformation to both character count and image count (log(n char) + log(n images)) to reduce skewness. This helps capture the distinction between emails with different character counts and image counts. Generated Variables: 'image_heavy' (int type), 'log_image_char' (float type).

- *Feature Selection* : We used **XGBoost** to select relevant features from the large number of variables after feature engineering, retaining only those with importance greater than 0.001 (around 10% of total features), mitigating overfitting risk. Lasso Regression and Mutual Information yielded unsatisfactory results compared to our final method.


### 2. Model Tuning and Comparison

#### 2.1. Model Tuning

After feature engineering, we explored different models to evaluate the performance. We splitted the training set with test size = 0.25. We then performed GridSearch to tune the parameters of our models. We evaluate the model by using repeated classified k fold cross-validation, with three repeats and 10 folds.

```python
# Obtaining average performance of the model by using repeated classified k fold cross-validation, with three repeats and 10 folds.
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(xg_top, x_top_feat, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```
We tested : 
- KNN
- RandomForest
- SVM
- XGBoost
- Gradient Boosting

We used KNN as our baseline model, initially achieving an accuracy of 0.4284 with n_neighbors = 3. Tuning n_neighbors to 30 improved accuracy to 0.5152, and using a Lasso model for feature selection further improved it to 0.5155. Cross-validation yielded an accuracy of 0.51993 on the Kaggle test set.

Our best performing model was XGBoost, initially achieving an accuracy of 0.605 without tuning. After feature selection and parameter tuning using RandomizedSearchCV, we obtained an accuracy of 0.6102 on our own test set and 0.598 on the Kaggle test set. Addressing duplicates in the dataset also contributed to the score improvement.


#### 2.2. Results

| Model  | Train Set Score after tuning | Test Set Score |
| ------------- | ------------- | ------------- |
| KNN  | 0.5155 | 0.5199 |
| XGBoost  | 0.6102 | 0.5983 |
| XGBoost/Duplicates  | 0.8877 | 0.7031 |
| SVM | 0.5975  | 0.5744 |
| RadomForest | 0.6026 | 0.5933 |

</details>



# Model selection of machine learning based exoplanet identification on the Kepler dataset

The Kepler dataset contains data from NASA's Kepler Space Telescope, launched in 2009 to identify habitable exoplanets. This project uses machine learning classifiers to speed up exoplanet identification by performing model selection on Kaggle's Kepler Exoplanet Search Result dataset. Our approach contributes to the field by providing a machine learning-based solution and achieves comparable results to other contributors. Finally, we apply our model to predict exoplanet likelihood for "Candidate" entries in the dataset.

## Context 
Machine learning classifiers are widely used for exoplanet identification, especially in the context of the NASA Kepler mission. Kepler records astrological objects but cannot confirm exoplanet status, requiring time-consuming observations by other instruments. Machine learning classifiers offer a more efficient approach for large-scale exoplanet detection by analyzing transit patterns. However, limited available data for training and testing, with only around 7000 data points, may limit the generalizability of classification models. To assess accuracy, we use a test-train split, but confirmation by NASA is necessary to gauge the model's generalizability. Further information can be found in the link below: 
https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results

## Data

This dataset is a cumulative record of all observed Kepler "objects of interest" — basically, all of the approximately 10,000 exoplanet candidates Kepler has taken observations on.

This dataset has an extensive data dictionary, which can be accessed here. Highlightable columns of note are:

- `kepoi_name`: A KOI is a target identified by the Kepler Project that displays at least one transit-like sequence within Kepler time-series photometry that appears to be of astrophysical origin and initially consistent with a planetary transit hypothesis
- `kepler_name` : [These names] are intended to clearly indicate a class of objects that have been confirmed or validated as planets—a step up from the planet candidate designation.
- `koi_disposition` : The disposition in the literature towards this exoplanet candidate. One of `CANDIDATE`, `FALSE POSITIVE`, `NOT DISPOSITIONED` or `CONFIRMED`.
- `koi_pdisposition` : The disposition Kepler data analysis has towards this exoplanet candidate. One of `FALSE POSITIVE`, `NOT DISPOSITIONED`, and `CANDIDATE`.
- `koi_score` : A value between 0 and 1 that indicates the confidence in the KOI disposition. For `CANDIDATEs`, a higher value indicates more confidence in its disposition, while for `FALSE POSITIVEs`, a higher value indicates less confidence in that disposition.

## Code

### 1. Methodology

#### 1.1. Data Exploration and Preprocessing

We analyzed the dataset using "info()" and "isna().sum()" and found it contains 9564 entries with 48 independent variables and one dependent variable, `koi_deposition`. `koi_deposition` has three values: "confirmed", "candidate", and "false positive". The project objective is to create a model that categorizes "candidate" bodies as "confirmed" or "false positive". This restricts our dataset to only "confirmed" and "false positive" entries, leaving us with a smaller sample of 7316 entries for model training, as we exclude "candidate" entries.

- *Data Type Analysis* : We examined the data types of the 48 independent variables. Four of them are of object type (`kepoi_name`, `kepler_name`, `koi_pdisposition`, `koi_tce_delivname`), and four are binary numerical (0 and 1), while the rest are continuous numerical variables. Since our modeling objective prioritizes numerical variables over categorical ones, we will need to transform or discard the object type variables for further processing. we decided to discard `kepoi_name`, `kepler_name`, `rowid` and `kepid` are to be considered as identification variables and therefore they do not bring any added value to the predictive capabilities of the model. We decided to also drop `koi_pdisposition` as it did not provide any additional information compared to `koi_diposition`.

- *Null Values Analysis*: We analyzed null values in the dataset, and identified that variables `koi_teq_err1` and `koi_teq_err2` had 100% missing values, so we decided to drop them. The variable `koi_score` had a high percentage of missing values (16%), but due to its strong correlation with the KOI disposition, filling the missing values with mean or mode could bias the model. Thus, we dropped this column as well. For other variables with less than 5% missing values, we filled them using appropriate statistics such as mode for `koi_tce_delivname` and median for other continuous variables, as the distributions had strong outliers positively skewing the mean.

- *Correlation Analysis*: We examined the correlation (figure below) between the remaining variables and identified pairs of variables with a strong correlation (|corr| > 0.85), mostly in the format of `koi_x_err1` and `koi_x_err2`. These variables represent the upper and lower bounds of the confidence interval for the measurement variable `koi_x`.

<img src="(https://github.com/ClaraBsnd/FML/blob/main/Project/corr.png?raw=true" width="48">


#### 1.2. Data Engineering 

- *Encoding* : we inspected `koi_tce_delivname` and noticed that it was made up by three possible values (87% of the occurrences being "q1_q17_dr25_tce") and as a consequence we deemed appropriate to one-hot encode it.


