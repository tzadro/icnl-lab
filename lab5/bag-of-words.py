import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from re import sub
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

def compute_accuracy(Y_, Y):
    num_correct = sum([y_ == y for y_, y in zip(Y_, Y)])
    num_total = len(Y_)
    
    return num_correct / num_total

def find_most_frequent(paths, N):
    dictionary = {}
    
    for path in paths:
        with open(path, 'r', encoding = 'latin-1') as file:
            for line in file:
                for word in line.split(' '):
                    clean_word = sub(r'\W+', '', word).lower()  # clean word
                    
                    if clean_word == '':  # if it's empty skip
                        continue
                    
                    if clean_word not in dictionary:  # if it's first time create new entry
                        dictionary[clean_word] = 0
                    
                    dictionary[clean_word] += 1  # update occurence
    
    by_frequency = sorted(dictionary.items(), reverse=True, key=lambda x: x[1])  # sort descending
    most_frequent = map(lambda x: x[0], by_frequency[:N])  # take only first N words
    return list(most_frequent)

def calculate_features(paths, most_frequent):
    X = []
    Y_ = []
    
    N = len(most_frequent)
    for path in paths:  # for every document (example)
        label = path.split('_')[-1]  # take label
        Y_.append(label)  # add it to labels
        
        total = 0  # track total number of words in document
        features = [0] * N  # feature vector size is determined by number of most frequent words we track
        
        with open(path, 'r', encoding = 'latin-1') as file:
            for line in file:
                for word in line.split(' '):
                    total += 1  # update total number of words
                    clean_word = sub(r'\W+', '', word).lower()  # clean word
                    
                    if clean_word not in most_frequent:  # if it is not one of the most frequent continue
                        continue
                    
                    features[most_frequent.index(clean_word)] += 1  # update occurence
        
        features = [f / total for f in features]  # divide every number of occurences by total number of words to get percentage
        X.append(features)  # add feature vector to list of examples
    
    return np.array(X), np.array(Y_)

def fit_and_compute_accuracy(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)  # fit
    Y = model.predict(X_test)  # predict
    return compute_accuracy(Y_test, Y)  # evaluate

if __name__ == 'main':
	colors = ['red', 'green', 'blue', 'magenta', 'orange', 'purple', 'cyan']
	data_dir = 'dataset'
	Ns = [5, 10, 20, 50, 100, 150, 200, 300]
	n_splits = 10

	data_paths = [data_dir + '/' + file for file in listdir(data_dir)]

	most_frequent = find_most_frequent(data_paths, Ns[-1])
	X, Y_ = calculate_features(data_paths, most_frequent)

	models = []
	models.append((BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5), 'Bagging'))
	models.append((RandomForestClassifier(n_estimators=10), 'Random Forest'))
	models.append((AdaBoostClassifier(n_estimators=100), 'AdaBoost'))
	models.append((GaussianNB(), 'Gaussian NB'))
	models.append((MultinomialNB(), 'Multinomial NB'))
	models.append((SVC(), 'SVC'))
	models.append((LinearSVC(), 'Linear SVC'))

	accuracies = np.empty((len(models), len(Ns), n_splits))
	for i, N in enumerate(Ns):  # for every N
	    kf = KFold(n_splits=n_splits)  # create k-fold
	    for j, (train, test) in enumerate(kf.split(X, Y_)):  # done n_k times where n_k is number of folds
	        X_train, Y_train = X[train,:N], Y_[train]  # take only N columns of feature vectors
	        X_test, Y_test = X[test,:N], Y_[test]  # same as above
	        
	        for k, (model, _) in enumerate(models):  # for every model check accuracy
	            accuracies[k][i][j] = fit_and_compute_accuracy(model, X_train, Y_train, X_test, Y_test)
	accuracies = np.sum(accuracies, axis=2) / n_splits  # sum all accuracies for every N for every model and divide by number of splits to get average

	for i, (_, name) in enumerate(models):
	    plt.plot(Ns, accuracies[i], label=name, c=colors[i])
	plt.title('Gender Prediction')
	plt.xlabel('N')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()