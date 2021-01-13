# Model process

import numpy as np
from keras.models import load_model
from SeqSelfAttention import *
T = 0.5

def LoadModel(scale, it):
    model = load_model('BaseModel\\' + scale + '-' + str(it) + '.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
    return model

def GSort(G2scale, G3scale, G4scale, cp2scale, cp3scale, cp4scale):
    G1, G2, G3, cp1, cp2, cp3, Gm, cpm = G2scale, G3scale, G4scale, cp2scale, cp3scale, cp4scale, 0.0, 0.0
    if G1 < G2:
        Gm = G1
        G1 = G2
        G2 = Gm
        cpm = cp1
        cp1 = cp2
        cp2 = cpm
    if G1 < G3:
        Gm = G1
        G1 = G3
        G3 = Gm
        cpm = cp1
        cp1 = cp3
        cp3 = cpm
    if G2 < G3:
        Gm = G2
        G2 = G3
        G3 = Gm
        cpm = cp2
        cp2 = cp3
        cp3 = cpm
    return G1, G2, G3, cp1, cp2, cp3

def GCalculation(cp):
    return abs(2 * cp - 1)

def GreedyFuzzyDecision(Score2, Score3, Score4):
    NumberSample = len(Score2)
    ScoreE = np.zeros((NumberSample, 2), float)
    for IndexSample in range(NumberSample):
        cp2scale, cp3scale, cp4scale = Score2[IndexSample][1], Score3[IndexSample][1], Score4[IndexSample][1]
        G2scale, G3scale, G4scale = GCalculation(cp2scale), GCalculation(cp3scale), GCalculation(cp4scale)
        G1, G2, G3, cp1, cp2, cp3 = GSort(G2scale, G3scale, G4scale, cp2scale, cp3scale, cp4scale)
        cpE = cp1
        if G1 < T:
            cpE = (cp1 + cp2) / 2
            Gnew = 2 * cpE - 1
            if Gnew < T:
                cpE = (cp1 + cp2 + cp3) / 3
        ScoreE[IndexSample][0] = 1 - cpE
        ScoreE[IndexSample][1] = cpE
    return ScoreE

def AddLabel(ScoreE, ListPair, it):
    Output = []
    Output.append('The ' + str(it + 1) + ' st/nd/rd/th independent prediction result \n')
    for indresult in range(len(ScoreE)):
        Sample = ListPair[indresult]
        miRNAname, lncRNAname, miRNAsequence, lncRNAsequence, miRNAstructure, lncRNAstructure = Sample.split(',')
        if ScoreE[indresult][1] >= 0.5:
            StrResult = miRNAname + ',' + lncRNAname + ',' + 'Interaction\n'
        else:
            StrResult = miRNAname + ',' + lncRNAname + ',' + 'Non-Interaction\n'
        Output.append(StrResult)
    Output.append('\n')
    return Output