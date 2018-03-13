# reads words and tags from a file
def load_dataset(input_file):
    X = [] # list of words
    Y_ = [] # list of corresponding tags (for word from X matched by index)
    
    with open(input_file, 'r', encoding = 'latin-1') as file:
        for line in file:
            example = line.replace('\n', '') # compensate for /n/r line endings
            elements = example.split('\t') # tab separated values
            word = elements[0]
            tag = elements[1]
            
            X.append(word)
            Y_.append(tag)
    
    return X, Y_

# writes words and predicted tags to file
# X - list of words
# Y - predicted tags for corresponding words
def save_predictions(output_file, X, Y):
    with open(output_file, 'w') as file:
        for word, tag in zip(X, Y):
            file.write(word + '\t' + tag + '\n')
            
# calculates accuracy
# Y_ - list of correct tags
# Y - list of predicted tags (matched by index)
def accuracy(Y_, Y):
    num_correct = sum([y_ == y for y_, y in zip(Y_, Y)]) # count number of equal elements at every index
    num_total = len(Y_)
    
    return num_correct / num_total # number of correct predictions / total number of examples

# class that stores releveant information for a word
class Word:
    def __init__(self):
        self.occurrences = {} # key: tag, value: number of examples tagged by tag in key
        self.count = 0 # total number of examples
    
    # record that word was tagged with tag count times
    def add_occurrences(self, tag, count=1):
        if tag not in self.occurrences: # if there is no record of an example tagged by tag
            self.occurrences[tag] = 0 # then create new
        
        self.occurrences[tag] += count
        self.count += count
    
    # returns list of tuples: (number of examples tagged by tag, tag)
    def get_occurrences(self):
        occrncs = []
        
        for tag in self.occurrences:
            occrncs.append((self.occurrences[tag], tag))
        
        return occrncs
    
    # returns list of tuples: (P(word|tag), tag)
    def get_probabilities(self):
        probs = []
        
        for tag in self.occurrences:
            prob = self.occurrences[tag] / self.count # P(word|tag) = count(word, tag) / count(word)
            probs.append((prob, tag))
        
        return probs

# class represents unigrams model
class Unigrams:
    def __init__(self):
        self.dictionary = {} # key: word, value: Word class tracking tag occurrences for word key

    # train model
    # X - list of words
    # Y_ - correct tags (tag at index i of Y_ corresponds to word at index i of X)
    def train(self, X, Y_):
        for word, tag in zip(X, Y_):
            if word not in self.dictionary: # if there is no tracker for word
                self.dictionary[word] = Word() # then create new

            self.dictionary[word].add_occurrences(tag) # for every training example record occurrence
    
    # saves model to a file as tsv
    # format: word tag number_of_occurrences
    def save(self, output_file):
        with open(output_file, 'w') as file:
            for word in self.dictionary:
                for num_occurrences, tag in self.dictionary[word].get_occurrences():
                    file.write(word + '\t' + tag + '\t' + str(num_occurrences) + '\n')
    
    # loads model from a file that has format like one in self.save()
    def load(self, input_file):
        self.dictionary = {}
        
        with open(input_file, 'r') as file:
            for line in file:
                elements = line.split('\t')
                word = elements[0]
                tag = elements[1]
                num_occurrences = int(elements[2])
                
                if word not in self.dictionary:
                    self.dictionary[word] = Word()
                    
                self.dictionary[word].add_occurrences(tag, num_occurrences)
    
    # for list of words X returns predictions Y (tag Y[i] for word X[i])
    def predict(self, X):
        Y = []
        
        for word in X:
            if word not in self.dictionary: # if there is no record for given word
                Y.append('NO_DATA') # then we're in trouble
                continue
            
            probabilities = self.dictionary[word].get_probabilities() # get tag probabilities from word counter
            tag = max(probabilities)[1] # select tag that has the biggest probability
            # ⌃can we replace this by random.choice() with prob distribution matching tag probabilities?⌃
            Y.append(tag)
        
        return Y

if __name__ == '__main__':
    TRAIN_SET = 'corpus.txt'
    MODEL_DATA = 'lexic.txt'
    TEST_SET_1 = 'gold_standard_1.txt'
    TEST_OUTPUT_1 = 'test_1_predictions.txt'
    TEST_SET_2 = 'gold_standard_2.txt'
    TEST_OUTPUT_2 = 'test_2_predictions.txt'

    # init model
    model = Unigrams()

    # train model
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