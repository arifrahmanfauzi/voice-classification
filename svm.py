import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, confusion_matrix,plot_confusion_matrix
import matplotlib.pyplot as plt

# Load data from numpy file
X =  np.load('feat.npy')
y =  np.load('label.npy').ravel()
print(y)
# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

# Simple SVM
print('fitting...')
clf = SVC(C=4, gamma=0.0001)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("accuracy=%0.3f" % acc)

# Grid search for best parameters
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 2, 3, 4, 5, 10, 15 ,20, 25,30,40,50]}]
#                     #  ,
#                     # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


scores = ['precision', 'recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print('')
#     # cv = cross validation
#     clf = GridSearchCV(SVC(C=4, gamma=0.0001), tuned_parameters, cv=3,
#                        scoring='%s_macro' % score)
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print('')
#     print(clf.best_params_)
#     print('')
#     print("Grid scores on development set:")
#     print('')
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print('')

print("Detailed classification report:")
print('')
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print('')
y_true, y_pred = y_test, clf.predict(X_test)
predicted = clf.predict(X_test)

print("hasilnya....... %s"%predicted)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
# print(matthews_corrcoef(y_true, y_pred))
print(y_true, y_pred)
# print(f1_score(y_true, y_pred))
print('')
titles_options = [("Confusion matrix, without normalization", None),
            ("Normalized confusion matrix", 'true')]
    
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf.fit(X_train, y_train), X_test, y_test,
                                 display_labels=y,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

# plt.show()
