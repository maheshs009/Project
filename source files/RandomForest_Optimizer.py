from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd
training_data = pd.read_csv("../Project/Data/feature_data.csv")[:2500]
training_results = pd.read_csv("../Project/Data/results.csv")[:2500]
test_data = pd.read_csv("../Project/Data/feature_data.csv")[2501:]
test_results = pd.read_csv("../Project/Data/results.csv")[2501:]
training_results = training_results['Adj Close'].values
test_results = test_results['Adj Close'].values

feature_variables = ['S&P Open','S&P High','S&P Low','S&P Close','S&P Volume','S&P Adj Close','LIBOR 1-Month','LIBOR 1-week','LIBOR 3-Month','LIBOR 6-month','LIBOR 12-Month','Overnight rate','VIX Open','VIX High','VIX Low','VIX Close','BKI Open','BKI High','BKI Low','BKI Close','BKI Volume','BKI Adj Close']

for variable in feature_variables:
    training_data[variable].fillna(training_data[variable].mean(), inplace=True)

training_data.drop(["Date"], axis=1, inplace=True)

#convert all ccategorical variables mentioned below to booleans of 0 and 1
categorical_variables = ['Month', 'Weekday']

for variable in categorical_variables:
    training_data[variable].fillna("Missing", inplace="True")
    dummies = pd.get_dummies(training_data[variable], prefix=variable)
    training_data = pd.concat([training_data, dummies], axis=1)
    training_data.drop([variable], axis=1, inplace=True)

#Feature importance
model = RandomForestRegressor(1000, oob_score=True, n_jobs=-1, random_state=42)
model.fit(training_data, training_results)
feature_importances = pd.Series(model.feature_importances_, index=training_data.columns)
feature_importances.sort()
feature_importances.plot(kind="barh", figsize=(12,12));

#n_estimators
results=[]
n_estimator_options = [30, 50, 100, 200, 300, 500, 700, 1000]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
    model.fit(training_data, training_results)
    print trees, "trees"
    score = model.score(training_data, training_results)
    print "Model Score: ", score
    print "oob_score is: ", model.oob_score_
    results.append(score)
    print ""

pd.Series(results, n_estimator_options).plot();

#max_features
results = []
max_features_options = ["auto", None, "sqrt", "log2", 0.9]

for max_features in max_features_options:
    model = RandomForestRegressor(n_estimators=800, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
    model.fit(training_data, training_results)
    print max_features, "option"
    score = model.score(training_data, training_results)
    print "Model Score: ", score
    print "oob_score is: ", model.oob_score_
    results.append(score)
    print""

pd.Series(results, max_features_options).plot(kind="barh", xlim=(.99,1.00));

#min_samples_leaf
results = []
min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for min_samples in min_samples_leaf_options:
    model = RandomForestRegressor(n_estimators=800, oob_score=True, 
                                  n_jobs=-1, random_state=42, max_features="sqrt", 
                                  min_samples_leaf=min_samples)
    model.fit(training_data, training_results)
    print min_samples, "min samples"
    score = model.score(training_data, training_results)
    print "Model Score: ", score
    print "oob_score is: ", model.oob_score_
    results.append(score)
    print""

pd.Series(results, min_samples_leaf_options).plot();

#Throughput

%%timeit
model = RandomForestRegressor(n_estimators=800, oob_score=True, 
                                  n_jobs=1, random_state=42, max_features="sqrt", 
                                  min_samples_leaf=1)
model.fit(training_data, training_results)

#RandomForest Regression
model = RandomForestRegressor(n_estimators=800, oob_score=True, 
                                  n_jobs=-1, random_state=42, max_features="sqrt", 
                                  min_samples_leaf=1)
model.fit(training_data, training_results)
model.score(training_data, training_results)

#Prepare test data
feature_variables = ['S&P Open','S&P High','S&P Low','S&P Close','S&P Volume','S&P Adj Close','LIBOR 1-Month','LIBOR 1-week','LIBOR 3-Month','LIBOR 6-month','LIBOR 12-Month','Overnight rate','VIX Open','VIX High','VIX Low','VIX Close','BKI Open','BKI High','BKI Low','BKI Close','BKI Volume','BKI Adj Close']

for variable in feature_variables:
    test_data[variable].fillna(test_data[variable].mean(), inplace=True)

test_data.drop(["Date"], axis=1, inplace=True)

categorical_variables = ['Month', 'Weekday']

for variable in categorical_variables:
    test_data[variable].fillna("Missing", inplace="True")
    dummies = pd.get_dummies(test_data[variable], prefix=variable)
    test_data = pd.concat([test_data, dummies], axis=1)
    test_data.drop([variable], axis=1, inplace=True)

print test_data.shape
print test_results.shape

#Final socre
print test_data.shape
print test_results.shape
model.score(test_data, test_results)

#Predicted file
savetxt('../Project/Data/predict.csv', model.predict(test_data), delimiter=',', fmt='%f')