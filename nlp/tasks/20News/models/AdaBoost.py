import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, StratifiedKFold
# import scikitplot as skplt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as ModelPipeline
import pandas as pd
from IPython.display import display

# from NewsGroup_common import newsgroup_data
# from NewsGroup_common import saveDataframe
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

############### Method Definition ###################
def basic_model_test(model,X_train,X_test,y_train,y_test,name):
    
    print("model fitting started")

    # fit the model
    model.fit(X_train, y_train)

    # model prediction
    print("Starting model Prediction")
    predictions = model.predict(X_test)

    # evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("\n"+name+" Model Classification Accuracy Score:", accuracy)

    # Classification report
    target_names = []
    for i in range(20):
        target_names.append(str(i))
    report = classification_report(y_test, predictions, target_names=target_names )
    print("\nClassification Report:", report)

    # confusion matrix
    confusion_matrix = get_confusion_matrix(y_test, predictions, labels=range(20))
    print(name+" Model Confusion Matrix: \n", confusion_matrix)

    return predictions, accuracy, report, confusion_matrix



############### Data Processing #####################

# X_train, X_test, y_train, y_test = newsgroup_data.getData()
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

############### AdaBoost Model #####################

adaboost = AdaBoostClassifier(n_estimators=50,learning_rate=1.0)
# predictions, accuracy, report, confusion_matrix = basic_model_test(adaboost,X_train, X_test, y_train, y_test,"NewsGroup AdaBoost")

################ Cross Validation Hyperparametre Tuning ###############################

print("\nHYPERPARAMETRE TUNING")
hyperparams = {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]}

optimized_model = GridSearchCV(estimator=adaboost, param_grid=hyperparams,
                                n_jobs=1, cv=3, verbose=1, error_score=1)

optimized_model.fit(X_train, y_train)

print(">>>>> Optimized params")
print(optimized_model.best_params_)

cv_results = optimized_model.cv_results_
print(">>>>>> Display the top results of grid search cv")
results_dataframe = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
display( results_dataframe.head() )
# saveDataframe(results_dataframe,"NewsGroup AdaBoost")

prediction = optimized_model.predict(X_test)
score = np.mean(prediction == y_test)
print("Using our training-dataset optimized adaboost model on the testing dataset for evaluating")
print("score = %f" % score)

print(metrics.classification_report(y_test, prediction))

################ Graph Reporting ###############################

# plot the confusion matrix
# skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=False, figsize=(12, 8))
# plt.show()