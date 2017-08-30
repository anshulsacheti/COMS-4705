Part I

1) Simple Enc-Dec Model (final_nmt_dynet)
2) Simple Att Model created with BLEU score 23.37 (gru_d=50) (final_nmt_dynet_attention)
                                            24.59      (gru_d=100) (final_nmt_dynet_attention_dbl)

Extra Credit:

Lookup params (nmt_dynet_attention_withLookUp)
Generated 50d word vectors for Chinese via instructions at https://github.com/Kyubyong/wordvectors
Downloaded globe.6B.zip for 50d word vectors for English from https://nlp.stanford.edu/projects/glove/
Score: N/A
Word Vector model for chinese and english data sets are 21MB and 171MB respectively. Iterating even 4000
samples takes at least 10 min. Could not generate a model.

Beam search (nmt_dynet_attention_beam)
Implemented
Score:14.027436, epoch: 7, beam_size = 5
Not enough time to run to completion

Best Score: 25.34, final_nmt_dynet_doubledSize model/embeddings, used nmt_dynet with gru_d=100


Part II

Other Preprocessing Steps:
One method we could use is converting all words like "don't" or "wouldn't" to "do not"
and "would not" so we have more uniformity in our data set. We also can take any and all
abbreviations and either expand them to their individual pieces or reduce them to the
abbreviation such as U.S.A to United States of America and vice-versa.

Practical Concerns:
One practical concern was defining the correct method to backtrack the graph.
Due to the recursive nature of the algorithm I had to be careful in how that was
implemented to make sure it didn't hit a recursive loop. Moreover, being careful
that the edges were represented correctly as they were referenced, namely [i,j]
is an edge from j to i and not vice versa. I think comparing a variety of decoder
options such as Eisner's vs a LSTM method would be great to compare the ability
of each to do it's job. And how much a vanilla LSTM model improves upon the Eisner
method.

English LAS, UAS:
Pos_d=0 :
UAS: 64.45
LAS: 54.23

Pos_d=25:
UAS: 75.65
LAS: 72.52

Swedish:
UAS: 80.54
LAS: 72.36

Korean:
UAS: 66.97
LAS: 55.76

Files all generated and in directory
