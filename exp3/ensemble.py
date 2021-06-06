import pickle
import numpy
from sklearn import tree
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit
        self.accuracy = []
        
       
        self.G      = [None for _ in range(n_weakers_limit)]
        self.alpha  = [  0  for _ in range(n_weakers_limit)]
        self.N      = 0
        self.detectionRate = 0.

        # true positive rate
        self.tpr = 0.
        # false positive rate
        self.fpr = 0.

        self.th  = 0.

    def is_good_enough(self):
        '''Optional'''
        output = self.predict(self._mat, self.th)
        correct = numpy.count_nonzero(output == self._label)/(self.samplesNum*1.)
        self.accuracy.append(correct)
        self.detectionRate = numpy.count_nonzero(output[0:self.posNum] == 1) * 1./ self.posNum

        Num_tp = 0 # Number of true positive
        Num_fn = 0 # Number of false negative
        Num_tn = 0 # Number of true negative
        Num_fp = 0 # Number of false positive
        for i in range(self.samplesNum):
            if self._label[i] == 1:
                if output[i] == 1:
                    Num_tp += 1
                else:
                    Num_fn += 1
            else:
                if output[i] == 1:
                    Num_fp += 1
                else:
                    Num_tn += 1

        self.tpr = Num_tp * 1./(Num_tp + Num_fn)
        self.fpr = Num_fp * 1./(Num_tn + Num_fp)

        if self.tpr > 0.999 and self.fpr < 0.0005:
            return True

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self._mat=X  
        self._label=y
        self.posNum = numpy.count_nonzero(self._label == 1)
        self.negNum = numpy.count_nonzero(self._label == -1)
        #pos_W = [1.0/(2 * self.posNum) for i in range(self.posNum)]
        #neg_W = [1.0/(2 * self.negNum) for i in range(self.negNum)]
        #init w
        self.w=numpy.ones(y.size)/float(y.size)
        
        
        for m in range(self.n_weakers_limit):
            self.N += 1
            self.G[m] = self.weak_classifier.fit(X, y)
            err_m = 1-self.G[m].score(X,y)
            
            # Alpha
            beta=err_m/(1-err_m)
            if(beta==0):
                self.alpha[m]=1
            else:
                self.alpha[m]= 0.5 * numpy.log(1/beta)
            
            output = self.G[m].predict(X)
            if(err_m!=0):
                for i in range(y.size):
                    if self._label[i] == output[i]:
                        self.w[i] *=  beta
                self.w/=sum(self.w)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        sampleNum = X.shape[0]

        output = numpy.zeros(sampleNum, dtype = numpy.float16)

        for i in range(self.N):
            output += self.G[i].predict(X) * self.alpha[i]

        return output


    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        output = self.predict_scores(X)
        for i in range(len(output)):
            if output[i] > threshold:
                output[i] = 1
            else:
                output[i] = -1

        return output
        
    def showErrRates(self):
        from matplotlib     import pyplot
        pyplot.title("The changes of accuracy (Figure by Jason Leaster)")
        pyplot.xlabel("Iteration times")
        pyplot.ylabel("Accuracy of Prediction")
        pyplot.plot([i for i in range(self.N)], 
                    [self.G[i].score(self._mat,self._label) for i in range(self.N)], '-.', 
                    label = "Accuracy * 100%")
        pyplot.axis([0., self.N, 0, 2.])
        pyplot.savefig("accuracyflow.jpg")

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
        
        
        
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
    save(classification_report(y_test, object.predict(X_test)),"classifier_report.txt")
    object.showErrRates()
    
learning(5)
