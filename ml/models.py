from sklearn.naive_bayes import MultinomialNB , GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

MODELS={
    "MultinomialNB":MultinomialNB(),
    "LogisticRegression":LogisticRegression(max_iter=1000),
    "LinearSVC":LinearSVC(),
    "KNN":KNeighborsClassifier(),
    "DecisionTree":DecisionTreeClassifier(max_depth=20),
    "RandomForest":RandomForestClassifier(n_estimators=50),
    "GaussianNB":GaussianNB()
}
