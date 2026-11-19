import numpy as np
import text_functions as tf
import nltk

#@author: The first version of this code is the courtesy of Vadim Selyanik

threshold = 15000 # Frequency threshold in the corpus ??
dimensions = [1000, 4000, 10000] # Dimensionality for high-dimensional vectors
lemmatizer = nltk.WordNetLemmatizer()  # create an instance of lemmatizer
ones_number = 2 # number of nonzero elements in randomly generated high-dimensional vectors
window_size = 2 #number of neighboring words to consider both back and forth. In other words number of words before/after current word
# zero_vector = np.zeros(dimension)
test_name = "new_toefl.txt" # file with TOEFL dataset
data_file_name = "lemmatized.text" # file with the text corpus

amount_dictionary = {}

# Count how many times each word appears in the corpus
text_file = open(data_file_name, "r")
for line in text_file:
    if line != "\n":
        words = line.split()
        for word in words:
            if amount_dictionary.get(word) is None:
                amount_dictionary[word] = 1
            else:
                amount_dictionary[word] += 1
text_file.close()

for dimension in dimensions:
    accuracies = []

    for run in range(1, 6):
        np.random.seed(run)
        zero_vector = np.zeros(dimension)
        
        dictionary = {} #vocabulary and corresponing random high-dimensional vectors
        word_space = {} #embedings




        #Create a dictionary with the assigned random high-dimensional vectors
        text_file = open(data_file_name, "r")
        for line in text_file: #read line in the file
            words = line.split() # extract words from the line
            for word in words:  # for each word
                if dictionary.get(word) is None: # If the word was not yed added to the vocabulary
                    if amount_dictionary[word] < threshold:
                        dictionary[word] = tf.get_random_word_vector(dimension, ones_number) # assign a  
                    else:
                        dictionary[word] = np.zeros(dimension) # frequent words are assigned with empty vectors. In a way they will not contribute to the word embedding

        text_file.close()


        #Note that in order to save time we only create embeddings for the words needed in the TOEFL task

            #Find all unique words amongst TOEFL tasks and initialize their embeddings to zeros    
        number_of_tests = 0
        text_file = open(test_name, "r") #open TOEFL tasks
        for line in text_file:
                words = line.split()
                words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                        words] # lemmatize words in the current test
                word_space[words[0]] = np.zeros(dimension)
                word_space[words[1]] = np.zeros(dimension)
                word_space[words[2]] = np.zeros(dimension)
                word_space[words[3]] = np.zeros(dimension)
                word_space[words[4]] = np.zeros(dimension)
                number_of_tests += 1
        text_file.close()




        # Processing the text to build the embeddings
        from collections import deque # List optimized for adding/removing at both ends
        from itertools import islice

        with open(data_file_name, "r") as text_file:
            # Read words continuously, regardless of line breaks
            tokens = (word for line in text_file for word in line.split())

            # Store words before and after the current word
            previous = deque(maxlen=window_size)
            upcoming = deque(islice(tokens, window_size + 1))

            while upcoming:
                target_word = upcoming[0]

                # Only build embeddings for TOEFL words
                if target_word in word_space:

                    # Shift word vectors to the left by -1 and add to target
                    for context in reversed(previous):
                        word_space[target_word] += np.roll(dictionary[context], -1)

                    # Shift word vectors to the right by +1 add to target
                    for context in islice(upcoming, 1, None):
                        word_space[target_word] += np.roll(dictionary[context], 1)

                # Move forward one word
                previous.append(upcoming.popleft())

                next_word = next(tokens, None)
                if next_word is not None:
                    upcoming.append(next_word)


        #Testing of the embeddings on TOEFL
        a = 0.0 # accuracy of the encodings    
        i = 0
        text_file = open(test_name, 'r')
        right_answers = 0.0 # variable for correct answers
        number_skipped_tests = 0.0 # some tests could be skipped if there are no corresponding words in the vocabulary extracted from the training corpus
        while i < number_of_tests:
                line = text_file.readline() #read line in the file
                words = line.split()  # extract words from the line
                words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                        words]  # lemmatize words in the current test
                try:
                    
                    if not(amount_dictionary.get(words[0]) is None): # check if there word in the corpus for the query word
                        k = 1
                        while k < 5:
                            # if amount_dictionary.get(words[k]) is None:
                            #     word_space[words[k]] = np.random.randn(dimension)
                            if np.array_equal(word_space[words[k]], zero_vector): # if no representation was learnt assign a random vector
                                word_space[words[k]] = np.random.randn(dimension)
                            k += 1
                        right_answers += tf.get_answer_mod([word_space[words[0]],word_space[words[1]],word_space[words[2]],
                                    word_space[words[3]],word_space[words[4]]]) #check if word is predicted right
                except KeyError: # if there is no representation for the query vector than skip
                    number_skipped_tests += 1
                    print("skipped test: " + str(i) + "; Line: " + str(words))
                except IndexError:
                    print(i)
                    print(line)
                    print(words)
                    break
                i += 1
        text_file.close()
        accuracy = 100 * right_answers / number_of_tests
        accuracies.append(accuracy)

        print(f"Dimension {dimension}, Run {run}: {accuracy:.2f}%")
    print(f"Average for dimension {dimension}: {np.mean(accuracies):.2f}%")
    print(f"Standard deviation: {np.std(accuracies):.2f}%")
    print()


