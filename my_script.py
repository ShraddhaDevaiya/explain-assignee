import pandas as pd
import numpy as np
from PyExplainer.pyexplainer import pyexplainer_pyexplainer


my_data = pd.read_csv('2bug_report_data.csv')

x = my_data['Priority']

y = my_data['Developer']

# step 1.1 - For the simplicity, we load the sample DataFrame that is included in the package already
sam_def = my_data

# step 1.2 - Define index column (OPTIONAL) and drop unwanted columns
sam_def = sam_def.set_index(sam_def['Issue_id'])
sam_def = sam_def.drop(['Issue_id'], axis=1)

# step 1.3 - Define feature cols (X), and label col (y)
# method to reduce number of features
from PyExplainer.pyexplainer.pyexplainer_pyexplainer import AutoSpearman
# select all rows, and all feature cols
# the last col, which is label col, is not selected

X = sam_def.iloc[:, :-1]
total_features = len(X.columns)

# apply feature selection function to our feature DataFrame
X = AutoSpearman(X)
selected = len(X.columns)

# select all rows, and the last label col
y = sam_def.iloc[:, -1]

print(selected, " out of ", total_features, " were selected via AutoSpearman feature selection process")
print('feature cols:', '\n\n', X, '\n\n')
print('label col:', '\n\n', y)

# step 1.4 - Split data into training and testing set
from sklearn.model_selection import train_test_split
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# step 2.1 - Train a RandomForest model using sklearn
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
rf_model.fit(X_train, y_train)

# step 2.2 - generate predictions
# generate prediction from the model, which will return a list of predicted labels
y_preds = rf_model.predict(X_test) 
# create a DataFrame which only has predicted label column
y_preds = pd.DataFrame(data={'PredictedReport': y_preds}, index=y_test.index) 
print("DBG : Y_PRED =  ", y_preds)

# step 3 - prediction post processing
# step 3.1 - Combine feature cols, label col, and the predicted col in testing set
combined_testing_data = X_test.join(y_test.to_frame())
combined_testing_data = combined_testing_data.join(y_preds)
combined_testing_data.head(3)
# total num of rows
total_rows = len(combined_testing_data)
print("DBG : COMBINED TEST DATA = ", combined_testing_data)

correctly_predicted_bug = combined_testing_data

# step 3.4 - Define feature cols and label col using correctly predicted testing data
# select all rows and feature cols
feature_cols = correctly_predicted_bug.iloc[:, :-2]
# selected all rows and one label col (either RealBug or PredictedBug is fine since they are the same)
label_col = correctly_predicted_bug.iloc[:, -2]

# step 3.5 - Select one row of correctly predicted bug to be explained
# decide which row to be selected
selected_row = 0
# select the row in X_test which contains all of the feature values
X_explain = feature_cols.iloc[[selected_row]]
# select the corresponding label from the DataFrame that we just created above
y_explain = label_col.iloc[[selected_row]]
print('one row of feature:', '\n\n', X_explain, '\n')
print('one row of label:', '\n\n', y_explain)

# 4. Create rules (explanations) and visualise it !
# step 4.1 - Initialise a PyExplainer object

py_explainer = pyexplainer_pyexplainer.PyExplainer(X_train = X_train,
                                                   y_train = y_train,
                                                   indep = X_train.columns,
                                                   dep = 'Developer',
                                                   blackbox_model = rf_model)

# step 4.2 - Create rules by triggering explain function under PyExplainer object
# Attention: This step can be time-consuming
rules = py_explainer.explain(X_explain=X_explain,
                             y_explain=y_explain,
                             search_function='crossoverinterpolation')

# Those created rules are stored in a dictionary, for more information about what is contained in each key, please refer to 'Appendix' part
rules.keys()

# step 4.3 - Simply trigger visualise function under PyExplainer object to visualise the created rules
py_explainer.visualise(rules)