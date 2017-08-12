Anshul Sacheti as5105
1)

a) Added "Figure_en.png" and "figure_sw.png". Also "Figure_en_nonProjective" as
per the piazza rubric.
b) A valid dependency tree is projective if for every arc (i,l,j) there is a
path from i to k for all i < k < j. If this is not true, then there must exist
an arc that overlaps another as otherwise there exists some word that does not
have a parent which is illegal.

2)

a) Implemented 4 ops

b) Had to convert a line in Sklearn in version 18.2 to make badFeatures.model work
Ran the Swedish test set on badFeatures.model.
Swedish:

UAS: 0.2793053545586107
LAS: 0.17872648335745298

As we can see from the UAS/LAS, the model does not do well in generating a
dependency tree using badFeatures.model. This is for a variety of reasons.
It doesn't take into account Lemmas, CTag, or fine-grained tags. It limits
itself to the first word of the buffer and stack when there is extra information
available through future values in the buffer or values 1 level deep in the
stack. We can see in part 3 that by adding this information, our model LAS/UAS
increases dramatically.

3)

a) Implemented right arc/shift operation

b)

The features added were Lemma, CTAG, and TAG for more values.


The implementation for each new feature was very simple. Looking at the feature
names above, it queried the BUF or STK for an index (if an index existed at that
location in the stack or buffer) which was O(1). It used that index to determine
the token associated with that index (O(1)). Then it checked whether the value
for the feature (FORM, LEMMA, TAG) existed (O(1)). And finally added that value
to a list of strings (O(1)).

Because each was simply done via querying either the stack or buffer and subsequently
executing some if statements, every new feature added had complexity O(1).

For the performance of each of these features, we use a baseline on the english
train dataset using the labelled dev data to provide a UAS/LAS. We first provide
a baseline with all features enabled, and then subsequently disable one at a time.
Thus each line indicates the feature that was disabled and how it impacted the final
UAS/LAS.

All enabled: UAS: 0.7887530562347188 LAS: 0.7569682151589242
BUF[0]_FORM: UAS: 0.7882640586797066 LAS: 0.7584352078239609
BUF[0]_LEMMA: UAS: 0.789242053789731 LAS: 0.7589242053789731
BUF[1]_FORM: UAS: 0.7887530562347188 LAS: 0.7579462102689487
BUF[0]_CTAG: UAS: 0.7863080684596577 LAS: 0.7540342298288508
BUF[0]_TAG: UAS: 0.7819070904645476 LAS: 0.7408312958435208
BUF[1]_TAG: UAS: 0.6601466992665037 LAS: 0.6278728606356968
BUF[2]_TAG: UAS: 0.7960880195599022 LAS: 0.7657701711491443
BUF[0]_FEATS: UAS: 0.7897310513447433 LAS: 0.7594132029339853
BUF[0]_LDEP: UAS: 0.7882640586797066 LAS: 0.7520782396088019
BUF[0]_RDEP: UAS: 0.7897310513447433 LAS: 0.7594132029339853
STK[0]_FORM: UAS: 0.7941320293398533 LAS: 0.7471882640586797
STK[0]_LEMMA: UAS: 0.7897310513447433 LAS: 0.7584352078239609
STK[0]_TAG: UAS: 0.7902200488997555 LAS: 0.7569682151589242
STK[0]_CTAG: UAS: 0.7902200488997555 LAS: 0.7550122249388753
STK[1]_TAG: UAS: 0.7867970660146699 LAS: 0.7525672371638141
STK[0]_FEATS: UAS: 0.7902200488997555 LAS: 0.7589242053789731
STK[0]_LDEP: UAS: 0.7921760391198044 LAS: 0.7613691931540343
STK[0]_RDEP: UAS: 0.7765281173594132 LAS: 0.7403422982885085

Given the variety of features added in this model, the best
combination comes from using all features listed above but BUF[2]_TAG. But more
importantly, we can see that most individual features' impact is negligible save
for BUF[1]_TAG. This could be for a variety of reasons, but it could be that
the information provided about a word in it's immediate context is most valuable
to defining it's dependencies. Thus the POS of a word adjacent to another in the
buffer can provide much more insight about the final tag. Moreover, due to the
variety of features available in this model, they can be similar in aspects such
as in the case of the TAG and CTAG values. This can lead to redundant features
and why we don't see many features heavily impacting our final UAS/LAS. This is
less of a possibility for BUF[1]_TAG because though other words can help infer
the tag of BUF[1]_TAG, they can't reliably always act as a stand in for it.

The features themselves focus on a variety of aspects of a given sentence.

The FORM feature focuses on the word itself. This provides the actual word encountered at
the current point in the sentence. This might not impact the final LAS/UAS because the combination
of other features paint an accurate description of the word itself.

Lemma provides a generalization of a word by returning its Lemma which can be helpful
for words that have different suffixes for example. Unfortunately for many words this
can also be blank, which can lead to a lack of information for the final solution.

CTAG provides a coarse tag that complements the fine tag provided by TAG. This might
be why we don't see a drastic difference in removing CTAG/TAG as they each provide
information similar to what the other would provide. Also extending to more words
in the buffer and the stack means that any given dependency tag has more context
to work with.

FEATS focuses on specific features and attributes like gender. As with lemma this
can also be empty and thus can have a negligible impact on our final LAS/UAS as many
words could have not have any extra features.

LDEP and RDEP represent the left and right dependencies for a given node in the
tree. LDEP/RDEP interestingly seem to have slightly different impacts. The
removal of LDEP slightly improves our results for both the BUF LDEP and STK LDEP,
while the removal of RDEP for both slightly worsens our results. In this case,
it could be that the LDEP is less strongly defined then RDEP in our training set,
and thus when trying to use LDEP as a feature it only adds noise to our model.

c) Models trained and saved as 'swedish.model' and 'english.model'

d)

Due to English and Swedish having inherently different dependency parses because
of how words are connected, the final solutions for both have slightly different
feature optimizations as noted below.

Swedish (using all features)
UAS: 0.816931982633864
LAS: 0.7221418234442837

English (using all features but BUF[2]_TAG)
UAS: 0.7965770171149145
LAS: 0.767237163814181

e. In your README.txt file, discuss in a few sentences the complexity of the
    arc-eager shift-reduce parser, and what tradeoffs it makes.

    As per the Nivre paper, “A Dynamic Oracle for Arc-Eager Dependency Parsing”,
    the arc-eager shift-reduce parser has a linear time complexity.

    A trade-off for this performance centers around projectivity. The arc-eager
    shift-reduce parser is only sound and complete for projective dependency
    forests as per the paper. This projectivity limitation prevents us from
    maintaining both soundness and completeness for non-projective languages of
    which there are many. This severely restricts the variety of languages that
    can be used and taken advantage of using this algorithm.


4)

a) Created parse.py
b) Done.
c) englishfile.conll generated
The output seems to show all words being derived from the root. Though this is
not as informative as the trees we have seen in the test data, these are still
valid.
