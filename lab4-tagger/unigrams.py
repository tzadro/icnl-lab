def load_dataset(input_file):
    X = []
    Y_ = []
    
    with open(input_file, 'r', encoding = 'latin-1') as file:
        for line in file:
            example = line.replace('\n', '')
            elements = example.split('\t')
            word = elements[0]
            tag = elements[1]
            
            X.append(word)
            Y_.append(tag)
    
    return X, Y_

def save_predictions(output_file, X, Y):
    with open(output_file, 'w') as file:
        for word, tag in zip(X, Y):
            file.write(word + '\t' + tag + '\n')
            
def accuracy(Y_, Y):
    num_correct = sum([y_ == y for y_, y in zip(Y_, Y)])
    num_total = len(Y_)
    
    return num_correct / num_total

class Word:
    def __init__(self):
        self.occurences = {}
        self.count = 0
    
    def add_occurences(self, tag, count=1):
        if tag not in self.occurences:
            self.occurences[tag] = 0
        
        self.occurences[tag] += count
        self.count += count
    
    def get_occurences(self):
        occrncs = []
        
        for tag in self.occurences:
            occrncs.append((self.occurences[tag], tag))
        
        return occrncs
    
    def get_probabilities(self):
        probs = []
        
        for tag in self.occurences:
            prob = self.occurences[tag] / self.count
            probs.append((prob, tag))
        
        return probs

class Unigrams:
    def __init__(self):
        self.dictionary = {}

    def train(self, X, Y_):
        for word, tag in zip(X, Y_):
            if word not in self.dictionary:
                self.dictionary[word] = Word()

            self.dictionary[word].add_occurences(tag)
    
    def save(self, output_file):
        with open(output_file, 'w') as file:
            for word in self.dictionary:
                for num_occurences, tag in self.dictionary[word].get_occurences():
                    file.write(word + '\t' + tag + '\t' + str(num_occurences) + '\n')
    
    def load(self, input_file):
        self.dictionary = {}
        
        with open(input_file, 'r') as file:
            for line in file:
                elements = line.split('\t')
                word = elements[0]
                tag = elements[1]
                num_occurences = int(elements[2])
                
                if word not in self.dictionary:
                    self.dictionary[word] = Word()
                    
                self.dictionary[word].add_occurences(tag, num_occurences)
    
    def predict(self, X):
        Y = []
        
        for word in X:
            if word not in self.dictionary:
                Y.append('NO_DATA')
                continue
            
            probabilities = self.dictionary[word].get_probabilities()
            tag = max(probabilities)[1]
            Y.append(tag)
        
        return Y

if __name__ == '__main__':
    TRAIN_SET = 'corpus.txt'
    MODEL_DATA = 'lexic.txt'
    TEST_SET_1 = 'gold_standard_1.txt'
    TEST_OUTPUT_1 = 'test_1_predictions.txt'
    TEST_SET_2 = 'gold_standard_2.txt'
    TEST_OUTPUT_2 = 'test_2_predictions.txt'

    model = Unigrams()

    # train
    X, Y_ = load_dataset(TRAIN_SET)
    model.train(X, Y_)

    # save model
    model.save(MODEL_DATA)
    # model.load(MODEL_DATA)

    # test_1
    X, Y_ = load_dataset(TEST_SET_1)
    Y = model.predict(X)
    save_predictions(TEST_OUTPUT_1, X, Y)
    print('test_1 accuracy: {:.2f}'.format(accuracy(Y_, Y)))

    # test_2
    X, Y_ = load_dataset(TEST_SET_2)
    Y = model.predict(X)
    save_predictions(TEST_OUTPUT_2, X, Y)
    print('test_2 accuracy: {:.2f}'.format(accuracy(Y_, Y)))