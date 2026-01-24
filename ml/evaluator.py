from sklearn.metrics import accuracy_score
def evaluate(models,X_train,y_train,X_test,y_test):
    scores={}
    trained={}
    for name , model in models.items():
        if name in ["KNN", "DecisionTree", "RandomForest", "GaussianNB"]:
            model.fit(X_train.toarray(),y_train)
            preds=model.predict(X_test.toarray())
        else:
            model.fit(X_train,y_train)
            preds=model.predict(X_test)
        scores[name]=accuracy_score(y_test , preds)
        trained[name]=model
    best=max(scores,key=scores.get)
    return best , trained[best],scores[best]
