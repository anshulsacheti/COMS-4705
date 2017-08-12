import numpy as np
import sys
from collections import defaultdict, namedtuple
from operator import itemgetter
import pdb

def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.

    scores - an (n+1) x (n+1) matrix
    gold - the gold arcs
    '''

    #YOUR IMPLEMENTATION GOES HERE
    #raise NotImplementedError

    #TODO: Handle Loss augmented inference
    #If golden provided, add/subtract(?) 1 to/from(?) each valid location (in scores? or in chart?)
    #Probably not in chart as the future values are just overriden. Using scores they are maintained always
    #raise NotImplementedError

    if gold:
        for i in range(len(gold)):
            scores[i,gold[i]]+=1

    #Directions
    # "left" = 0
    # "right" = 1

    #Completeness
    incomplete = False
    complete = True

    #Table dimension
    m = scores.shape[0]
    # print("Table Size: %d x %d" % (m, m))
    #DP Table
    chart = {}

    #backtrack table
    backtrack = {}

    #Initialize Table
    for i in range(m):
        chart[i,i,"left",complete]      = 0
        chart[i,i,"right",complete]     = 0
        chart[i,i,"left",incomplete]    = -np.inf
        chart[i,i,"right",incomplete]   = -np.inf

    #Eisner algorithm with backtracking
    for l in range(1,m):
        for i in range(0,m-l):
            j = i + l

            #Find max in each range
            max1=-np.inf
            max2=-np.inf
            max3=-np.inf
            max4=-np.inf
            k1=-np.inf
            k2=-np.inf
            k3=-np.inf
            k4=-np.inf

            # print("i val: %d, j val: %d, range: %s" % (i, j, list(range(i,j))))
            for k in range(i,j):

                tmp = chart[i,k,"right",complete] + chart[k+1,j,"left",complete] + scores[i,j]
                if tmp>max1:
                    max1=tmp
                    k1=k

            #Assign best values to each arc for this set of iterations
            chart[i,j,"right",incomplete]   =max1
            backtrack[i,j,"right",incomplete]   =k1

            for k in range(i+1,j+1):
                tmp = chart[i,k,"right",incomplete] + chart[k,j,"right",complete]
                if tmp>max2:
                    max2=tmp
                    k2=k

            #Assign best values to each arc for this set of iterations
            chart[i,j,"right",complete  ]   =max2
            backtrack[i,j,"right",complete  ]   =k2

            for k in range(i,j):
                tmp = chart[i,k,"right",complete] + chart[k+1,j,"left",complete] + scores[j,i]
                if tmp>max3:
                    max3=tmp
                    k3=k

            #Assign best values to each arc for this set of iterations
            chart[i,j,"left" ,incomplete]   =max3
            backtrack[i,j,"left" ,incomplete]   =k3

            for k in range(i,j):
                tmp = chart[i,k,"left",complete] + chart[k,j,"left",incomplete]
                if tmp>max4:
                    max4=tmp
                    k4=k

            #Assign best values to each arc for this set of iterations
            chart[i,j,"left" ,complete  ]   =max4
            backtrack[i,j,"left" ,complete  ]   =k4

            #Assign best values to each arc for this set of iterations
            # chart[i,j,d_right,incomplete]   =max1
            # chart[i,j,d_right,complete  ]   =max2
            # chart[i,j,d_left ,incomplete]   =max3
            # chart[i,j,d_left ,complete  ]   =max4
            #
            # backtrack[i,j,d_right,incomplete]   =k1
            # backtrack[i,j,d_right,complete  ]   =k2
            # backtrack[i,j,d_left ,incomplete]   =k3
            # backtrack[i,j,d_left ,complete  ]   =k4

    #Get parents for best path
    arcDict={}
    bt(backtrack,0,m-1,"right",complete,arcDict)

    #Generate list of parents
    keys = list(arcDict.keys())
    keys.sort()
    arcs = [arcDict[i] for i in keys]
    arcs.insert(0,-1)

    # if gold:
    #     print("Gold arcs: %s" % (gold))
    #     print("arcs     : %s" % (arcs))
    #     print("percent wrong: %f" % (sum(abs(np.array(gold)-np.array(arcs))!=0)/len(arcs)))
    # pdb.set_trace()
    return arcs

#Eisner Backtrack algorithm from lecture video
def bt(chart,i,j,d,c,h):

    #Empty arc, return
    if i == j:
        return

    #Get split point
    k = chart[i,j,d,c]

    #using eisner equations to backtrack steps
    if c==True:
        if d=="right":
            bt(chart,i,k,"right",False,h)
            bt(chart,k,j,"right",True,h)

        if d=="left":
            bt(chart,i,k,"left",True,h)
            bt(chart,k,j,"left",False,h)
    else:
        # print("backtrack: i:%d, j:%d" % (i,j))
        if d=="right":
            h[j]=i
        if d=="left":
            h[i]=j
        bt(chart,i,k,"right",True,h)
        bt(chart,k+1,j,"left",True,h)


    #Modify the algorithm to do loss augmented inference.
    #Given the gold standard arcs, we can encourage the parsing algorithm
    #to select the correct parse and avoid overfitting by adding 1 to the
    #scores of known incorrect arcs during training. More specifically,
    #add 1 to scores[i,j] if there is an arc from j to i.
