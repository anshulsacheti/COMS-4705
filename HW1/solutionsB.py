import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a
#   list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline
#   character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined
#   by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list
#   of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list
#   of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    #Iterate over lines, and split into word list and tag list
    #Add START_SYMBOL and STOP_SYMBOL
    for line in brown_train:
        #line = START_SYMBOL + " " + line + " " + STOP_SYMBOL
        splitLine = line.split()

        wordList = [START_SYMBOL, START_SYMBOL]
        tagList  = [START_SYMBOL, START_SYMBOL]

        #Break up each term into word/tag
        for term in splitLine:
            terms=term.split("/")
            tagList.extend([terms[-1]])
            #Separated by "/" earlier, add back any removed in other loc
            wordList.extend(["/".join(terms[:-1])])

        wordList.extend([STOP_SYMBOL])
        tagList.extend([STOP_SYMBOL])

        brown_words.append(wordList)
        brown_tags.append(tagList)

        # print(wordList)
        # print(tagList)
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}

    modified_corpus = []

    #Simplify to one string

    for line in brown_tags:
        modified_corpus.extend(line)

    #To calculate trigram probabilities, need unigram/bigram probabilities
    #Copied from part A

    fdU = nltk.FreqDist(modified_corpus)
    fdU_ref = fdU.copy()
    fdU_count = fdU.N() - 1.5*fdU[STOP_SYMBOL]
    for key in fdU.keys():
        fdU[key] = math.log(fdU[key]/fdU_count,2)

    # print("UNIGRAM: %f, %f, %f" % (fdU["captain"], fdU["captain's"], fdU["captaincy"]))

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
    tg = nltk.trigrams(modified_corpus)
    fdT = nltk.FreqDist(tg)

    for key in fdT.keys():
        firstKey = key[0]
        secondKey = key[1]
        q_values[key] = math.log(fdT[key]/fdB_ref[(firstKey, secondKey)],2)

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = list(q_values.keys())
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])

    modified_corpus = []

    #Simplify to one string
    for line in brown_words:
        modified_corpus.extend(line)

    #Get all words that occur more than RARE_WORD_MAX_FREQ
    bwFreq = nltk.FreqDist(modified_corpus)

    for key in bwFreq.keys():
        if bwFreq[key]>RARE_WORD_MAX_FREQ:
            known_words.add(key)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []

    #Iterate over each sentence
    #Replace any word not in known_words set with RARE_SYMBOL
    for sentence in brown_words:
        rSentence = sentence.copy()
        for i in range(len(sentence)):
            if sentence[i] not in known_words:
                rSentence[i] = RARE_SYMBOL
        #print(rSentence)
        brown_words_rare.append(rSentence)
    #print(brown_words_rare[0])
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])

    tag_word_dict = {}

    #Iterate over each sentence, pairing together word/tag
    #Generates reference dictionary for all words with a given tag
    for i in range(len(brown_words_rare)):
        sentence  = brown_words_rare[i]
        tags = brown_tags[i]
        for j in range(len(sentence)):
            word = sentence[j]
            tag = tags[j]

            #Populate dictionary with various keys and additions
            if tag not in tag_word_dict.keys():
                tag_word_dict[tag] = {word:1, "count":1}
            else:
                if word not in tag_word_dict[tag].keys():
                    tag_word_dict[tag][word]=1
                else:
                    tag_word_dict[tag][word]+=1
                tag_word_dict[tag]["count"]+=1

    #Calculate log emission probabilities
    #Generate python dict with word/tag tuple keys
    for tag in tag_word_dict.keys():
        taglist.add(tag)
        for word in tag_word_dict[tag]:
            wordCount = tag_word_dict[tag][word]
            tagCount = tag_word_dict[tag]["count"]
            e_values[(word, tag)] = math.log(wordCount/tagCount,2)

    assert e_values[("America", "NOUN")]==-10.999343168270888
    #America NOUN -10.99925955

    try:
        assert e_values[("Columbia", "NOUN")]==-13.560058122745367
    except AssertionError:
        print(e_values[("Columbia", "NOUN")])
    #Columbia NOUN -13.5599745045

    try:
        assert e_values[("New", "ADJ")]==-8.18848005226213
    except AssertionError:
        print(e_values[("New", "ADJ")])
    #New ADJ -8.18848005226

    try:
        assert e_values[("York", "NOUN")]==-10.712061216190417
    except AssertionError:
        print(e_values[("York", "NOUN")])
    #York NOUN -10.711977598
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = list(e_values.keys())
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
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

    if a==LOG_PROB_OF_ZERO:
        return LOG_PROB_OF_ZERO

    if (b-a)<-20:
        return a
    else:
        return (a + math.log(1+math.exp(b-a),2))


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags
#   (taglist), a set of all known words (known_words), trigram probabilities
#   (q_values) and emission probabilities (e_values) and outputs a list where
#   every element is a tagged sentence (in the WORD/TAG format, separated by
#   spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the
#   words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG",
#   separated by spaces. Each sentence is a string with a terminal newline,
#   not a list of tokens. Remember also that the output should not contain the
#   "_RARE_" symbol, but rather the original words of the sentence!
def forward(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    #Iterate over all sentences
    #Generate ref table to calculate forward probabilities

    #print(taglist)
    for sentence in brown_dev_words:

        #Initializations
        forward_table = {}

        tmp = [START_SYMBOL, START_SYMBOL]
        tmp.extend(sentence)
        tmp.extend([STOP_SYMBOL])
        sentence = tmp

        e_keys = e_values.keys()
        q_keys = q_values.keys()

        rareTag = [RARE_SYMBOL]

        #Populate all instances for first trigram in sentence
        word = sentence[2]; t_1 = t_2 = '*'
        # print("Word: %s" % (word))
        if word not in known_words:
            word = RARE_SYMBOL

        for tag in taglist:

            #Confirm key valid, use RARE_SYMBOL as necessary
            #Ignore if emission prob==0
            if (word,tag) in e_keys:
                emission = e_values[(word,tag)]
            else:

                #Illegal emission, ignore as per problem definition
                continue
            # print("-->Emission with tag: %s is %f" % (tag, emission))
            #Assign tag trigram probability
            if (t_1,t_2,tag) not in q_keys:
                q_val = LOG_PROB_OF_ZERO
            else:
                q_val = q_values[(t_1,t_2,tag)]

            # print("  -->Trigram (*,*,%s) Prob: %f" % (tag, q_val))
            #Update tables
            forward_table[(0,t_2,tag)] = emission+q_val
            # print("    -->Assigned tag: %s with tag_seq: (*,*,%s) to word: %s with new Prob: %f" % (tag, tag, word, forward_table[(0,t_2,tag)]))
        #print("Starting vt_keys: %s" % (list(forward_table.keys())))

        #Iterate over all n^3 combination of tags to find best tag for word
        for i in range(1,len(sentence)-2):
            # print("Word: %s" % (sentence[i+2]))

            #Replace word with RARE_SYMBOL if required
            word = sentence[i+2]
            if word not in known_words:
                word = RARE_SYMBOL

            for tag1 in taglist:

                #Calculate emission
                #If none exists, continue onto next tag because just LOG_PROB_OF_ZERO
                if (word,tag1) in e_keys:
                    emission = e_values[(word,tag1)]
                else:

                    #By hw definition, ignore if prob==0
                    #print("Didn't find word: %s" % (word))
                    continue

                # print("-->Emission with tag: %s is %f" % (tag1, emission))
                #Generate all tag sets that could exist
                for tag3 in taglist:
                    for tag2 in taglist:
                        # Check if trigram tag legal
                        if (tag3,tag2,tag1) in q_keys:
                            tg_prob = q_values[(tag3,tag2,tag1)]
                        else:
                            tg_prob = LOG_PROB_OF_ZERO

                        #Calculate probability
                        #Ignore any previous tag sets that don't exist (as problem definition says to ignore them)
                        try:
                            vt_val = forward_table[(i-1, tag3, tag2)]
                        except KeyError:
                            continue

                        # print("  -->Trigram (%s,%s,%s) Prob: %f" % (tag3, tag2, tag1, tg_prob))
                        tmp = tg_prob + emission + vt_val

                        try:
                            lS = logSum(forward_table[(i, tag2, tag1)],tmp)
                            # print("    -->logSum: %f, old: %f, new: %f" % (lS, forward_table[(i, tag2, tag1)],tmp))
                            forward_table[(i, tag2, tag1)] = lS
                        except KeyError:
                            forward_table[(i, tag2, tag1)] = tmp
                            #print(bestTags)
                        # print("      -->Added tag: %s with tag_seq: (%s,%s,%s) to word: %s with prob: %f. New Prob: %f" % (tag1, tag3, tag2, tag1, word, tmp,forward_table[(i, tag2, tag1)]))

            #print("vt_keys: %s\n" % (list(forward_table.keys())))
                #for tag2 in taglist:

        #Find largest probability at last step
        #Iterate over forward_table and determine probability for final step
        lastStep=len(sentence)-3
        probSum = 0
        #print(sentence)
        #print(forward_table)
        #print(list(forward_table.keys()))
        for key in forward_table.keys():
            val = key[0]
            if val==lastStep and key[2]==STOP_SYMBOL:
                probSum += forward_table[key]
                #probSum += math.pow(2,forward_table[key])

        tagged.append(probSum)
        #print(probSum)

    outfile = open("output/B7.txt", 'w')
    for sentence in tagged:
        outfile.write(str(sentence)+'\n')
    outfile.close()

    return tagged

# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags
#   (taglist), a set of all known words (known_words), trigram probabilities
#   (q_values) and emission probabilities (e_values) and outputs a list where
#   every element is a tagged sentence (in the WORD/TAG format, separated by
#   spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the
#   words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG",
#   separated by spaces. Each sentence is a string with a terminal newline,
#   not a list of tokens. Remember also that the output should not contain the
#   "_RARE_" symbol, but rather the original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    #Iterate over all sentences
    #Generate ref table to calculate ideal tagged list

    #print(taglist)
    for sentence in brown_dev_words:

        #Initializations
        viterbi_table = {}
        backtrack_table = {}

        tmp = [START_SYMBOL, START_SYMBOL]
        tmp.extend(sentence)
        tmp.extend([STOP_SYMBOL])
        sentence = tmp

        e_keys = e_values.keys()
        q_keys = q_values.keys()

        rareTag = [RARE_SYMBOL]

        #Populate all instances for first trigram in sentence
        word = sentence[2]; t_1 = t_2 = '*'
        if word not in known_words:
            word = RARE_SYMBOL

        for tag in taglist:

            #Confirm key valid, use RARE_SYMBOL as necessary
            #Ignore if emission prob==0
            if (word,tag) in e_keys:
                emission = e_values[(word,tag)]
            else:

                #Illegal emission, ignore as per problem definition
                continue
            #Assign tag trigram probability
            if (t_1,t_2,tag) not in q_keys:
                q_val = LOG_PROB_OF_ZERO
            else:
                q_val = q_values[(t_1,t_2,tag)]

            #Update tables
            viterbi_table[(0,t_2,tag)] = emission+q_val
            backtrack_table[(0,t_2,tag)] = ("*", "*")
        #print("Starting vt_keys: %s" % (list(viterbi_table.keys())))

        #Iterate over all n^3 combination of tags to find best tag for word
        for i in range(1,len(sentence)-2):
            #print("Word: %s" % (sentence[i+2]))

            #Replace word with RARE_SYMBOL if required
            word = sentence[i+2]
            if word not in known_words:
                word = RARE_SYMBOL

            for tag1 in taglist:

                #Calculate emission
                #If none exists, continue onto next tag because just LOG_PROB_OF_ZERO
                if (word,tag1) in e_keys:
                    emission = e_values[(word,tag1)]
                else:

                    #By hw definition, ignore if prob==0
                    #print("Didn't find word: %s" % (word))
                    continue

                #print("-->Emission with tag: %s is %f" % (tag1, emission))
                maxProb = -math.inf
                bestTags = ()
                #Generate all tag sets that could exist
                for tag3 in taglist:
                    for tag2 in taglist:
                        # Check if trigram tag legal
                        if (tag3,tag2,tag1) in q_keys:
                            tg_prob = q_values[(tag3,tag2,tag1)]
                        else:
                            tg_prob = LOG_PROB_OF_ZERO

                        #print("------>Trigram (%s,%s,%s) Prob: %f" % (tag3, tag2, tag1, tg_prob))
                        #Calculate probability
                        #Ignore any previous tag sets that don't exist (as problem definition says to ignore them)
                        try:
                            vt_val = viterbi_table[(i-1, tag3, tag2)]
                        except KeyError:
                            continue
                        tmp = tg_prob + emission + vt_val

                        #print("  -->Trying to assign %f for tags %s %s %s" % (tmp, tag3, tag2, tag1))
                        if tmp>maxProb:
                            #print("    --> Assigned new tags")
                            maxProb = tmp
                            bestTags = (tag3, tag2, tag1)

                #Update table with best prob
                #If no valid options exist, don't add anything for this entry

                viterbi_table[(i, bestTags[1], tag1)] = maxProb
                backtrack_table[(i,bestTags[1],tag1)] = (bestTags[0],bestTags[1])
                #print(bestTags)
                #print("  -->Assigned tag: %s with tag_seq: (%s,%s,%s) to word: %s with prob: %f" % (tag1, bestTags[0], bestTags[1], bestTags[2], sentence[i+2], maxProb))

            #print("vt_keys: %s\n" % (list(viterbi_table.keys())))
                #for tag2 in taglist:

        #Find largest probability at last step
        #Iterate over viterbi_table and determine best probability at last step
        lastStep=len(sentence)-4
        bestSol = ()
        bestProb = -math.inf
        for key in viterbi_table.keys():
            val = key[0]
            if val==lastStep and viterbi_table[key]>bestProb:
                bestProb = viterbi_table[key]
                bestSol = key

        #Get tags for last words in most likely tag set
        try:
            lastKey = (bestSol[1], bestSol[2])
        except:
            print(sentence)
            print("lastStep: %d" % (lastStep))
            print(viterbi_table)

        #Create string for each sentence
        #Backtrace through table using best tags
        taggedSent = sentence[lastStep+2] + "/" + lastKey[1] + "\n"
        while  lastStep>0:
            lastKey=backtrack_table[(lastStep,lastKey[0],lastKey[1])]
            lastStep = lastStep-1
            taggedSent = sentence[lastStep+2] + "/" + lastKey[1] + " " + taggedSent

        #print(taggedSent)
        tagged.append(taggedSent)

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG",
# separated by spaces. Each sentence is a string with a terminal newline, not a
# list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in range(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []

    #Flatten zipped up training file
    modified_corpus = []
    for sentence in training:
        modified_corpus.append(list(sentence))

    #Create taggers with backoff
    taggerA = nltk.DefaultTagger('NOUN')
    taggerB = nltk.BigramTagger(modified_corpus,backoff=taggerA)
    taggerC = nltk.TrigramTagger(modified_corpus,backoff=taggerB)

    #Add start/stop symbols to brown_dev_words
    modified_corpus = []
    for line in brown_dev_words:
        tmp = [START_SYMBOL, START_SYMBOL]
        tmp.extend(line)
        tmp.extend([STOP_SYMBOL])
        modified_corpus.append(tmp)

    #Test tagger setup
    tagged_sents = taggerC.tag_sents(modified_corpus)

    #Convert tags into proper format
    for sentence in tagged_sents:
        sentence = sentence[2:-1]
        convertedStr = ""
        for i in range(len(sentence)):
            if i == (len(sentence)-1):
                convertedStr = convertedStr + str(sentence[i][0]) + "/" + str(sentence[i][1]) + "\n"
            else:
                convertedStr = convertedStr + str(sentence[i][0]) + "/" + str(sentence[i][1]) + " "
        tagged.append(convertedStr)
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do forward on brown_dev_words (question 5)
    forward_probs = forward(brown_dev_words, taglist, known_words, q_values, e_values)

    # do viterbi on brown_dev_words (question 6)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print("Part B time: " + str(time.clock()) + ' sec')

if __name__ == "__main__": main()
