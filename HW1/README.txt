as5105 Anshul Sacheti

Part A

1)
  UNIGRAM near -12.456068696432359
  BIGRAM near the -1.561878887608115
  TRIGRAM near the ecliptic -5.392317422778761

2) Perplexity of corpus for 3 different models

python perplexity.py output/A2.uni.txt data/Brown_train.txt
The perplexity is 1052.4865859
python perplexity.py output/A2.bi.txt data/Brown_train.txt
The perplexity is 53.8984761198
python perplexity.py output/A2.tri.txt data/Brown_train.txt
The perplexity is 5.7106793082

3)

python perplexity.py output/A3.txt data/Brown_train.txt
The perplexity is 12.5516094886

4) When you compare the performance (perplexity) between the best model without
    interpolation and the models with linear interpolation, is the result you
    got expected?

    The result is what I expected. The perplexity increased because we reduced the strength of a more accurate
    model (trigram) by interpolating it with less accurate models. Hence the final result has higher perplexity.

5)

python perplexity.py output/Sample1_scored.txt data/Sample1.txt
The perplexity is 11.1670289158

python perplexity.py output/Sample2_scored.txt data/Sample2.txt
The perplexity is 1.15079721592e+173

Sample1 more likely belongs to the Brown dataset, and Sample2 does not. Due to Sample2's magnitudes greater perplexity score our learned
models had either chose unigrams/bigrams/trigrams that were very unlikely or could not find a match. In either case, this points to
an input that does not share a similar vocabulary to our learned models and thus the Brown corpus.

Sample1 also has a perplexity that is within the range we've seen with our other learned models indicating that it is similar and more
likely part of the Brown Corpus.

Part B

2)

TRIGRAM CONJ ADV NOUN -4.466503667312799
TRIGRAM DET NUM NOUN -0.7132001285162394
TRIGRAM NOUN PRT CONJ -6.385032741040191

4)

* * 0.0
midnight NOUN -13.181546499491636
Place VERB -15.454122544938643
primary ADJ -10.066801495673879
STOP STOP 0.0
_RARE_ VERB -3.1775619067202294
_RARE_ X -0.5463596614974089

5)
#B7.txt used for output (B5 used in main def already)
Probability of first word sequence: -199.7366066518593

6)
#B5.txt as assigned in main def
python pos.py output/B5.txt data/Brown_tagged_dev.txt
Percent correct tags: 93.2393925772

7)
#B6.txt used as assigned in main def
python pos.py output/B6.txt data/Brown_tagged_dev.txt
Percent correct tags: 91.4413586882


Part A time: 14.069611 sec
Part B time: 255.941995 sec
