# Project: Classification of Iris Flowers
# Dataset Location: UCI Machine Learning Repository
# Link to Dataset: "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

# Created By Aaditya Raval on 08/27/2021

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Check shape
print(dataset.shape)

# Check starting data samples
print(dataset.head(20))

# Dataset descriptive statistics
print(dataset.describe())

# Class distribution for each label
print(dataset.groupby('class').size())

# Box and whisker (univariate) plots
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False,
             sharey=False)
pyplot.show()

# Histograms
dataset.hist()
pyplot.show()

# Scatter (multivariate) matrix
scatter_matrix(dataset)
pyplot.show()

# Train and validation split
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# Machine learning models
models = []
models.append(('LR', LogisticRegression(solver='liblinear',
                                        multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Cross validation
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,
                                 scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Plot model comparison
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluation metrics
print("Prediction Accuracy: ", accuracy_score(Y_validation, predictions))
print("Confusion Matrix: ")
print(confusion_matrix(Y_validation, predictions))
print("Classification Report: ")
print(classification_report(Y_validation, predictions))
