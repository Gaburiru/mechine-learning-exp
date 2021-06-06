import pickle
from sklearn import tree
from ensemble import AdaBoostClassifier

if __name__ == "__main__":
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)
    
    def load(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
    data=load('feature.txt')
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size=0.3, random_state=13)
    y_train=train_set[:,0]
    y_test=test_set[:,0]
    X_train=train_set[:,1:]
    X_test=test_set[:,1:]
    #print(y_train.shape,y_test.shape,X_train.shape,X_test.shape)
    from sklearn.metrics import classification_report
    def learning(n_iter):
        object=AdaBoostClassifier(tree.DecisionTreeClassifier(),n_iter)
        object.fit(X_train,y_train)
        #save(classification_report(y_test, object.predict(X_test)),"classifier_report.txt")
        object.showErrRates()
        
    learning(5)