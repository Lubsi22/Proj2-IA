# 2. Import libraries and modules
import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib 
 
# 3. Load Matosinhos fires data.
data = pd.read_csv("forest_fires.csv", sep=',')
 
# 4. Split data into training and test sets
y = data["area"]
X = data.drop('area', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    )
 
# 5. Declare data preprocessing steps
pipeline = make_pipeline(
                         GradientBoostingRegressor(
                                random_state=123
                                               ))
 
# 6. Declare hyperparameters to tune
hyperparameters = { 'gradientboostingregressor__n_estimators': [100, 200],
    'gradientboostingregressor__learning_rate': [0.01, 0.05, 0.1],
    'gradientboostingregressor__max_depth': [2, 3, 5],
    'gradientboostingregressor__subsample': [0.8, 1.0],
    'gradientboostingregressor__min_samples_split': [2, 5],
    'gradientboostingregressor__min_samples_leaf': [1, 2]
    }
 
# 7. Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=5)
 
clf.fit(X_train, y_train)
 
# 8. Refit on the entire training set
# No additional code needed if clf.refit == True (default is True)
 
# 9. Evaluate model pipeline on test data
pred = clf.predict(X_test)
print( r2_score(y_test, pred) )
print( mean_squared_error(y_test, pred) )

best_model = clf.best_estimator_.named_steps['gradientboostingregressor']

importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importances)
 
# 10. Save model for future use
joblib.dump(clf, 'gradient_boosting_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')