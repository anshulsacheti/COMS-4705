from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from providedcode.dependencygraph import DependencyGraph
from featureextractor import FeatureExtractor
from transition import Transition
import sys
import pdb

#example
#cat englishfile | python parse.py english.model > englishfile.conll

#Get param input
modelName = sys.argv[1]

#Load model
tp = TransitionParser.load(modelName)

#iterate over lines
#query using model
for line in sys.stdin:
    dg = DependencyGraph.from_sentence(line)
    p = tp.parse([dg])
    print(p[0].to_conll(10).encode('utf-8').decode())
