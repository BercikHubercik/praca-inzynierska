# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def MNB(X, y):
    vector = CountVectorizer()
    X_vec = vector.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    return model.predict(X_test)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
