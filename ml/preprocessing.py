from sklearn.feature_extraction.text import TfidfVectorizer
def create_vectorizer(train_text,test_text=None):
    vectorizer=TfidfVectorizer(max_features=3000,min_df=3)
    X_train=vectorizer.fit_transform(train_text)
    X_test=vectorizer.transform(test_text) if test_text is not None else None
    return vectorizer , X_train,X_test 
