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


## Code
The code for the email classification can be found in the email_classification.py file. 

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

