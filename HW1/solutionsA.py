import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens
# separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples
# expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):

    modified_corpus = []

    #Add start and stop strings to data
    #Simplify to one string
    for line in training_corpus:
        tmp = [START_SYMBOL]
        tmp.extend(line.split())
        tmp.extend([STOP_SYMBOL])
        modified_corpus.extend(tmp)

    #Get frequency counts of each set for each n-gram
    #Use math.log(x,2) to calculate log probability
    fdU = nltk.FreqDist(modified_corpus)
    fdU_ref = fdU.copy()
    fdU_count = fdU.N() - 1.0*fdU[STOP_SYMBOL]
    for key in fdU.keys():
        fdU[key] = math.log(fdU[key]/fdU_count,2)

    # print("UNIGRAM: %f, %f, %f" % (fdU["captain"], fdU["captain's"], fdU["captaincy"]))

    modified_corpus = []

    #Add start and stop strings to data
    #Simplify to one string (need slightly different string for bigram/trigram)
    for line in training_corpus:
        tmp = [START_SYMBOL, START_SYMBOL]
        tmp.extend(line.split())
        tmp.extend([STOP_SYMBOL])
        modified_corpus.extend(tmp)

    #Bigram
    bg = nltk.bigrams(modified_corpus)
    fdB = nltk.FreqDist(bg)
    fdB_ref = fdB.copy()

    for key in fdB.keys():
        firstKey = key[0]
        fdB[key] = math.log(fdB[key]/fdU_ref[firstKey],2)

    # print("BIGRAM: %f, %f, %f" % (fdB[("and", "religion")],
    #             fdB[("and", "religious")], fdB[("and", "religiously")]))

    #Trigram
    #Need new model to properly handle first word and last word

    #Add start and stop strings to data
    #Simplify to one string
    # for line in training_corpus:
    #     tmp = [START_SYMBOL, START_SYMBOL]
    #     tmp.extend(line.split())
    #     tmp.extend([STOP_SYMBOL, STOP_SYMBOL])
    #     modified_corpus.extend(tmp)

    tg = nltk.trigrams(modified_corpus)
    fdT = nltk.FreqDist(tg)

    for key in fdT.keys():
        firstKey = key[0]
        secondKey = key[1]
        fdT[key] = math.log(fdT[key]/fdB_ref[(firstKey, secondKey)],2)

    # print("TRIGRAM: %f, %f, %f" % (fdT[("and", "not", "a")],
    #             fdT[("and", "not", "by")], fdT[("and", "not", "come")]))
    #
    # print("UNIGRAM near: %f, BIGRAM near the: %f, TRIGRAM: near the ecliptic %f"
    #         % (fdU[("near")],fdB[("near", "the")], fdT[("near", "the", "ecliptic")]))
    unigram_p = dict(fdU)
    bigram_p = dict(fdB)
    trigram_p = dict(fdT)
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram,
# and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = list(unigrams.keys())
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = list(bigrams.keys())
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = list(trigrams.keys())
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens
# separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is
# the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    scores = []

    #Iterate over each line
    #Split line into tokens
    #Calculate per n-gram probability for each set of words in sentence, get product
    for line in corpus:

        #Update line to include correct number of START_SYMBOL/STOP_SYMBOL
        if n<2:
            line = START_SYMBOL + " " +line+ " " + STOP_SYMBOL
        else:
            line = START_SYMBOL + " " + START_SYMBOL + " " +line+ " " + STOP_SYMBOL
        splitLine = line.split()
        lineScore = 0

        for i in range(len(splitLine)-n+1):
            #print("i: %d" % (i))
            if n>1:
                val = tuple(splitLine[i:i+n])
            else:
                val = splitLine[i:i+n][0]
                if val==START_SYMBOL:
                    continue
            #print("val: %s" % (val,))
            #print("ngram_p[val]: %s" % (ngram_p[val]))
            try:
                prob = ngram_p[val]
            except KeyError:
                lineScore = MINUS_INFINITY_SENTENCE_LOG_PROB
                scores.append(lineScore)
                #print("Found missing val: %s" (val,))
                break
            lineScore = lineScore+prob
            #print("lineScore %s" % (lineScore))
        scores.append(lineScore)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


def logSum(a,b):
    """
        Calculates log(a+b), where a and b are both log(some_value)
        Log Trick based on lecture notes
        Returns: log(a+b)
        a: log(some_value)
        b: log(some_value)
    """

    #Change order such that a>=b for equation to work
    if b>a:
        tmp = b; b = a; a = tmp;

    if a==MINUS_INFINITY_SENTENCE_LOG_PROB:
        return MINUS_INFINITY_SENTENCE_LOG_PROB

    if (b-a)<-20:
        return a
    else:
        return (a + math.log(1+math.exp(b-a),2))

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly
# interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that
# express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []

    lambda1=lambda2=lambda3=1/3

    #Iterate over all sentences
    for i in range(len(corpus)):

        lineScore = 0
        #Create line
        line = corpus[i].split()
        tmp = [START_SYMBOL, START_SYMBOL]; tmp.extend(line); tmp.extend([STOP_SYMBOL])
        line = tmp

        # uniscores = score(unigrams, 1, [corpus[i]])
        # biscores = score(bigrams, 2, [corpus[i]])
        # triscores = score(trigrams, 3, [corpus[i]])
        #
        # tmpScore = logSum(logSum(uniscores[0],biscores[0]),triscores[0])
        # print("i: %f, ugs: %f, bgs: %f, tgs: %f" % (tmpScore, uniscores[0],
        #                                             biscores[0],triscores[0]))

        #Sum average of different models and take log
        for tg in nltk.trigrams(line):

            if tg[-1] in unigrams.keys():
                ugScore = unigrams[tg[-1]]
            else:
                ugScore = MINUS_INFINITY_SENTENCE_LOG_PROB
            if tg[1:] in bigrams.keys():
                bgScore = bigrams[tg[1:]]
            else:
                bgScore = MINUS_INFINITY_SENTENCE_LOG_PROB
            if tg in trigrams.keys():
                tgScore = trigrams[tg]
            else:
                tgScore = MINUS_INFINITY_SENTENCE_LOG_PROB

            #Handle case where can't find word
            if tgScore==MINUS_INFINITY_SENTENCE_LOG_PROB and bgScore==MINUS_INFINITY_SENTENCE_LOG_PROB and ugScore==MINUS_INFINITY_SENTENCE_LOG_PROB:
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                continue
            else:
                tmp = math.log(lambda1*(math.pow(2,ugScore)+math.pow(2,bgScore)+math.pow(2,tgScore)),2)
                lineScore += tmp
        scores.append(lineScore)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print("Part A time: " + str(time.clock()) + ' sec')

if __name__ == "__main__": main()
