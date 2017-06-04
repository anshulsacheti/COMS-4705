import nltk
from nltk.corpus import brown as brown

#Part 1a
if True==False:
  def findtags(tag_prefix, tagged_text):
      cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                    if tag.startswith(tag_prefix))
      return dict((tag, cfd[tag]) for tag in cfd.conditions())

  tagdict = findtags('NN', brown.tagged_words())
  nn = tagdict["NN"].most_common(1000)

  pluralSetSize=0
  nouns = []
  for (noun, count) in nn:
    plural = noun + "s"
    #print("Got here with noun %s and count %d and looking at plural %s" % (noun, count, plural))
    pluralCount = tagdict["NNS"][plural]
    if pluralCount > count:
      #print("%s with count %d vs %d" % (plural, pluralCount ,count))
      nouns.append(plural)
      pluralSetSize=pluralSetSize+1
      if pluralSetSize >= 5: break
  print("nouns: %s" % nouns)
#--------------------------------------------


#Part 1b
if True==False:
  maxTags=0
  gWord = ""
  #From NLTK Tutorial
  tagged = brown.tagged_words()
  data = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in tagged)
  for word in sorted(data.conditions()):
    tags = [tag for (tag, _) in data[word].most_common()]
    if len(tags)>maxTags:
      #print(word, ' '.join(tags))
      gWord = word
      gTags = ' '.join(tags)
      maxTags = len(tags)
  print("gword: %s with tags %s" % (gWord,gTags))


#---------------------------------

#Part 1c


if True==False:
  from nltk.corpus import brown
  #From NLTK Tutorial
  tagged = brown.tagged_words(tagset="universal")
  tag_fd = nltk.FreqDist(tag for (word, tag) in tagged)
  tag_freq=tag_fd.most_common()
  tag_freq.sort(key=lambda x: x[1], reverse=True)
  print("tag_freq: %s" % tag_freq)


#----------------------------------

#Part 2a

testString = ["The", "boat", "is", "going", "to", "sink", "and", "I", "am", "scared", "!"]
if True==False:
  from nltk.corpus import brown
  fd = nltk.FreqDist(brown.words())
  cfd = nltk.ConditionalFreqDist(brown.tagged_words())

  #Unigram
  most_freq_words = fd.most_common()
  likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
  unigram_tagger = nltk.UnigramTagger(model=likely_tags)
  print("unigram tagging: %s" % unigram_tagger.tag(testString))

  #Bigram
  bigram_tagger = nltk.BigramTagger(brown.tagged_sents())
  print("bigram tagging: %s" % bigram_tagger.tag(testString))

  #pos_tag
  print("pos tagging: %s" % nltk.pos_tag(testString))

#-----------------------------------
#Part 2b

# code goes below
testString = ["I", "had", "a", "dream", "that", "I", "found", "a", "lost", "dog", "and", "instead", "of", "taking",
              "it", "to", "its", "rightful", "owner", ",", "I", "brought", "it", "home", "and", "kept", "it", "."]

if True==False:
  from nltk.corpus import brown
  fd = nltk.FreqDist(brown.words())
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(tagset='universal'))

  #Referenced using NLTK Book

  #Unigram
  most_freq_words = fd.most_common()
  likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
  unigram_tagger = nltk.UnigramTagger(model=likely_tags)
  print("unigram tagging: %s\n" % unigram_tagger.tag(testString))

  #Bigram
  bigram_tagger = nltk.BigramTagger(brown.tagged_sents(tagset='universal'))
  print("bigram tagging: %s\n" % bigram_tagger.tag(testString))

  #pos_tag
  print("pos tagging: %s" % nltk.pos_tag(testString, tagset='universal'))

#-----------------------------------
#Part 2c
testString = ["I'm", "procrastinating", "my", "code", "for", "this", "assignment", "!"]
if True==True:
  from nltk.corpus import brown
  fd = nltk.FreqDist(brown.words())
  cfd = nltk.ConditionalFreqDist(brown.tagged_words())

  #Unigram
  most_freq_words = fd.most_common()
  likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
  unigram_tagger = nltk.UnigramTagger(model=likely_tags)
  print("unigram tagging: %s" % unigram_tagger.tag(testString))

  #Bigram
  bigram_tagger = nltk.BigramTagger(brown.tagged_sents())
  print("bigram tagging: %s" % bigram_tagger.tag(testString))

  #pos_tag
  print("pos tagging: %s" % nltk.pos_tag(testString))
