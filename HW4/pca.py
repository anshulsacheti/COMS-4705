import numpy as np
from sklearn.decomposition import PCA
import json
import pandas
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import utils
import graphParser


#Get data from train file and test file for verbs
train_file='data/english/train.conll'
vocabTrain = utils.Vocabulary(train_file)
train = list(utils.read_conll(train_file))
indices, pos_indices, gold_arcs, gold_labels = vocabTrain.process(train,deterministic=True)

word_embeddings_json = json.load(open('output/model_embeddings.json'))

# To access the word and POS embeddings, respectively, use ‘word’ and ‘pos’ as
# the key to the dictionary. This will produce another dictionary of answers
# to their embeddings, which you will need to convert to a matrix. Then run
# PCA on the embedding matrix (but make sure to keep track of which words
# match up with which embeddings)

word_embeddings=word_embeddings_json['word']

#Get size of matrix using json data
rowCount = len(list(word_embeddings.keys()))
colCount = len(word_embeddings[list(word_embeddings.keys())[0]])

#Create and populate matrix
word_embedding_matrix = np.zeros((rowCount, colCount))
for i,key in enumerate(word_embeddings.keys()):
    row = word_embeddings[key]
    word_embedding_matrix[i,:] = row

pca = PCA(n_components=2)

fitted_word_embeddings=pca.fit_transform(word_embedding_matrix)

#Valid verb POS
#'VBD', 'VB', 'VBZ', 'VBN', 'VBP', 'VBG'
verbsPOS = ['VBD', 'VB', 'VBZ', 'VBN', 'VBP', 'VBG']
verbs = []
verbEmbeddings = []

#Iterate over all words
#Use dict sets of pos_words to determine if word is a verb
for i,word in enumerate(word_embeddings.keys()):
    for verbType in verbsPOS:
        if word in vocabTrain.pos_words[verbType]:
            # print("Found %s as %s" % (word, verbType))
            verbs.append(word)
            if verbEmbeddings==[]:
                verbEmbeddings=np.array([fitted_word_embeddings[i]])
            else:
                # print("verbEmbed: %s, fitted: %s" % (verbEmbeddings, fitted_word_embeddings[i]))
                verbEmbeddings=np.append(verbEmbeddings, [fitted_word_embeddings[i]],axis=0)
            break

labels = verbs
wordFig=plt.figure(1)
plt.scatter(verbEmbeddings[:,0], verbEmbeddings[:,1])
for x, y, label in zip(verbEmbeddings[:, 0], verbEmbeddings[:, 1], labels):
    plt.annotate(label,xy=(x, y))
wordFig.show()

#=============================================================================
#POS

pos_embeddings_json = json.load(open('output/model_pos_embeddings.json'))
pos_embeddings=pos_embeddings_json['pos']

rowCount = len(list(pos_embeddings.keys()))
colCount = len(pos_embeddings[list(pos_embeddings.keys())[0]])

#Create and populate matrix
pos_embedding_matrix = np.zeros((rowCount, colCount))
for i,key in enumerate(pos_embeddings.keys()):
    row = pos_embeddings[key]
    pos_embedding_matrix[i,:] = row

pca = PCA(n_components=2)

fitted_pos_embeddings=pca.fit_transform(pos_embedding_matrix)

labels = list(pos_embeddings.keys())
posFig = plt.figure(2)
plt.scatter(fitted_pos_embeddings[:,0], fitted_pos_embeddings[:,1])
for x, y, label in zip(fitted_pos_embeddings[:, 0], fitted_pos_embeddings[:, 1], labels):
    plt.annotate(label,xy=(x, y))
posFig.show()
# Then plot the results using a plotting library such as matplotlib
# to make a scatterplot with each point represented on the scatterplot
# as its corresponding word.


# Create visualizations for the English parts-of-speech and verbs (a verb is
# any word that appears as a verb in the training data). Note that you should run
# PCA on all the word embeddings, but only plot the verbs. After you have created
# these visualizations, save them to files called pos_visualization.png and
# verb_visualization.png.

plt.show()
#Check if word is a verb in training set

try:
    wordIdx = vocabTrain.word2idx["word"]
    pos = vocabTrain.idx2pos[wordIdx]
except KeyError:
    pass
