# PmliPEMG for cross validation

import numpy as np
import re, math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, Convolution2D, MaxPooling2D, LSTM, Concatenate
from keras import optimizers
from sklearn.metrics import roc_auc_score
from SeqSelfAttention import *
# np.random.seed(1337) # repeated experiment
PathData = 'Example\\TrainingValidationSet.npy'
scale = [2, 5930, 3, 3954, 4, 2965]
separator, Sequencekmertotal, SequenceGgaptotal, Structurekmertotal, StructureGgaptotal = ' ', 3, 3, 3, 3

def SequencekmerExtract(sequence, totalkmer):
    sequence = sequence.replace('U', 'T')
    character = 'ATCG'
    sequencekmer = ''
    for k in range(totalkmer):
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalkmer - kk))
        # 1-mer
        if kk == 1:
            for char11 in character:
                s1 = char11
                f1 = wk * sequence.count(s1) / sk
                string1 = str(f1) + separator
                sequencekmer = sequencekmer + string1
        # 2-mer
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    s2 = char21 + char22
                    numkmer2 = 0
                    for lkmer2 in range(len(sequence) - kk + 1):
                        if sequence[lkmer2] == s2[0] and sequence[lkmer2 + 1] == s2[1]:
                            numkmer2 = numkmer2 + 1
                    f2 = wk * numkmer2 / sk
                    string2 = str(f2) + separator
                    sequencekmer = sequencekmer + string2
        # 3-mer
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        s3 = char31 + char32 + char33
                        numkmer3 = 0
                        for lkmer3 in range(len(sequence) - kk + 1):
                            if sequence[lkmer3] == s3[0] and sequence[lkmer3 + 1] == s3[1] and sequence[lkmer3 + 2] == s3[2]:
                                numkmer3 = numkmer3 + 1
                        f3 = wk * numkmer3 / sk
                        string3 = str(f3) + separator
                        sequencekmer = sequencekmer + string3
        # 4-mer
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            s4 = char41 + char42 + char43 + char44
                            numkmer4 = 0
                            for lkmer4 in range(len(sequence) - kk + 1):
                                if sequence[lkmer4] == s4[0] and sequence[lkmer4 + 1] == s4[1] and sequence[lkmer4 + 2] == s4[2] and sequence[lkmer4 + 3] == s4[3]:
                                    numkmer4 = numkmer4 + 1
                            f4 = wk * numkmer4 / sk
                            string4 = str(f4) + separator
                            sequencekmer = sequencekmer + string4
        # 5-mer
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            for char55 in character:
                                s5 = char51 + char52 + char53 + char54 + char55
                                numkmer5 = 0
                                for lkmer5 in range(len(sequence) - kk + 1):
                                    if sequence[lkmer5] == s5[0] and sequence[lkmer5 + 1] == s5[1] and sequence[lkmer5 + 2] == s5[2] and sequence[lkmer5 + 3] == s5[3] and sequence[lkmer5 + 4] == s5[4]:
                                        numkmer5 = numkmer5 + 1
                                f5 = wk * numkmer5 / sk
                                string5 = str(f5) + separator
                                sequencekmer = sequencekmer + string5
        # 6-mer
        if kk == 6:
            for char61 in character:
                for char62 in character:
                    for char63 in character:
                        for char64 in character:
                            for char65 in character:
                                for char66 in character:
                                    s6 = char61 + char62 + char63 + char64 + char65 + char66
                                    numkmer6 = 0
                                    for lkmer6 in range(len(sequence) - kk + 1):
                                        if sequence[lkmer6] == s6[0] and sequence[lkmer6 + 1] == s6[1] and sequence[lkmer6 + 2] == s6[2] and sequence[lkmer6 + 3] == s6[3] and sequence[lkmer6 + 4] == s6[4] and sequence[lkmer6 + 5] == s6[5]:
                                            numkmer6 = numkmer6 + 1
                                    f6 = wk * numkmer6 / sk
                                    string6 = str(f6) + separator
                                    sequencekmer = sequencekmer + string6
    return sequencekmer

def SequenceGgapExtract(sequence, totalGgap):
    sequence = sequence.replace('U', 'T')
    character = 'ATCG'
    sequenceGgap = ''
    for k in range(totalGgap):
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalGgap - kk))
        if kk == 1:
            for char11 in character:
                for char12 in character:
                    for char13 in character:
                        for char14 in character:
                            num1 = 0
                            for l1 in range(len(sequence) - kk - 3):
                                if sequence[l1] == char11 and sequence[l1 + 1] == char12 and sequence[l1 + kk + 2] == char13 and sequence[l1 + kk + 3] == char14:
                                    num1 = num1 + 1
                            f1 = wk * num1 / sk
                            string1 = str(f1) + separator
                            sequenceGgap = sequenceGgap + string1
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    for char23 in character:
                        for char24 in character:
                            num2 = 0
                            for l2 in range(len(sequence) - kk - 3):
                                if sequence[l2] == char21 and sequence[l2 + 1] == char22 and sequence[l2 + kk + 2] == char23 and sequence[l2 + kk +3] == char24:
                                    num2 = num2 + 1
                            f2 = wk * num2 / sk
                            string2 = str(f2) + separator
                            sequenceGgap = sequenceGgap + string2
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        for char34 in character:
                            num3 = 0
                            for l3 in range(len(sequence) - kk - 3):
                                if sequence[l3] == char31 and sequence[l3 + 1] == char32 and sequence[l3 + kk + 2] == char33 and sequence[l3 + kk + 3] == char34:
                                    num3 = num3 + 1
                            f3 = wk * num3 / sk
                            string3 = str(f3) + separator
                            sequenceGgap = sequenceGgap + string3
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            num4 = 0
                            for l4 in range(len(sequence) - kk - 3):
                                if sequence[l4] == char41 and sequence[l4 + 1] == char42 and sequence[l4 + kk + 2] == char43 and sequence[l4 + kk + 3] == char44:
                                    num4 = num4 + 1
                            f4 = wk * num4 / sk
                            string4 = str(f4) + separator
                            sequenceGgap = sequenceGgap + string4
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            num5 = 0
                            for l5 in range(len(sequence) - kk - 3):
                                if sequence[l5] == char51 and sequence[l5 + 1] == char52 and sequence[l5 + kk + 2] == char53 and sequence[l5 + kk + 3] == char54:
                                    num5 = num5 + 1
                            f5 = wk * num5 / sk
                            string5 = str(f5) + separator
                            sequenceGgap = sequenceGgap + string5
    return sequenceGgap

def SequenceGgapExtract2(sequence, totalGgap):
    sequence = sequence.replace('U', 'T')
    character = 'ATCG'
    sequenceGgap = ''
    for k in range(totalGgap):
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalGgap - kk))
        if kk == 1:
            for char11 in character:
                for char12 in character:
                    num1 = 0
                    for l1 in range(len(sequence) - kk - 1):
                        if sequence[l1] == char11 and sequence[l1 + kk + 1] == char12:
                            num1 = num1 + 1
                    f1 = wk * num1 / sk
                    string1 = str(f1) + separator
                    sequenceGgap = sequenceGgap + string1
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    num2 = 0
                    for l2 in range(len(sequence) - kk - 3):
                        if sequence[l2] == char21 and sequence[l2 + kk + 1] == char22:
                            num2 = num2 + 1
                    f2 = wk * num2 / sk
                    string2 = str(f2) + separator
                    sequenceGgap = sequenceGgap + string2
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    num3 = 0
                    for l3 in range(len(sequence) - kk - 3):
                        if sequence[l3] == char31 and sequence[l3 + kk + 1] == char32:
                            num3 = num3 + 1
                    f3 = wk * num3 / sk
                    string3 = str(f3) + separator
                    sequenceGgap = sequenceGgap + string3
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    num4 = 0
                    for l4 in range(len(sequence) - kk - 3):
                        if sequence[l4] == char41 and sequence[l4 + kk + 1] == char42:
                            num4 = num4 + 1
                    f4 = wk * num4 / sk
                    string4 = str(f4) + separator
                    sequenceGgap = sequenceGgap + string4
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    num5 = 0
                    for l5 in range(len(sequence) - kk - 3):
                        if sequence[l5] == char51 and sequence[l5 + kk + 1] == char52:
                            num5 = num5 + 1
                    f5 = wk * num5 / sk
                    string5 = str(f5) + separator
                    sequenceGgap = sequenceGgap + string5
    return sequenceGgap

def StructurekmerExtract(structure, totalkmer):
    character = ').'
    structurekmer = '' # 特征

    sp = structure.split()
    ssf = sp[0]
    ssf = ssf.replace('(', ')')
    for k in range(totalkmer):
        kk = k + 1
        sk = len(ssf) - kk + 1
        wk = 1 / (2 ** (totalkmer - kk))
        # 1-mer
        if kk == 1:
            for char11 in character:
                s1 = char11
                f1 = wk * ssf.count(s1) / sk
                string1 = str(f1) + separator
                structurekmer = structurekmer + string1
        # 2-mer
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    s2 = char21 + char22
                    numkmer2 = 0
                    for lkmer2 in range(len(ssf) - kk + 1):
                        if ssf[lkmer2] == s2[0] and ssf[lkmer2 + 1] == s2[1]:
                            numkmer2 = numkmer2 + 1
                    f2 = wk * numkmer2 / sk
                    string2 = str(f2) + separator
                    structurekmer = structurekmer + string2
        # 3-mer
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        s3 = char31 + char32 + char33
                        numkmer3 = 0
                        for lkmer3 in range(len(ssf) - kk + 1):
                            if ssf[lkmer3] == s3[0] and ssf[lkmer3 + 1] == s3[1] and ssf[lkmer3 + 2] == s3[2]:
                                numkmer3 = numkmer3 + 1
                        f3 = wk * numkmer3 / sk
                        string3 = str(f3) + separator
                        structurekmer = structurekmer + string3
        # 4-mer
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            s4 = char41 + char42 + char43 + char44
                            numkmer4 = 0
                            for lkmer4 in range(len(ssf) - kk + 1):
                                if ssf[lkmer4] == s4[0] and ssf[lkmer4 + 1] == s4[1] and ssf[lkmer4 + 2] == s4[2] and ssf[lkmer4 + 3] == s4[3]:
                                    numkmer4 = numkmer4 + 1
                            f4 = wk * numkmer4 / sk
                            string4 = str(f4) + separator
                            structurekmer = structurekmer + string4
        # 5-mer
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            for char55 in character:
                                s5 = char51 + char52 + char53 + char54 + char55
                                numkmer5 = 0
                                for lkmer5 in range(len(ssf) - kk + 1):
                                    if ssf[lkmer5] == s5[0] and ssf[lkmer5 + 1] == s5[1] and ssf[lkmer5 + 2] == s5[2] and ssf[lkmer5 + 3] == s5[3] and ssf[lkmer5 + 4] == s5[4]:
                                        numkmer5 = numkmer5 + 1
                                f5 = wk * numkmer5 / sk
                                string5 = str(f5) + separator
                                structurekmer = structurekmer + string5
    return structurekmer

def StructureGgapExtract(structure, totalGgap):
    character = ').'
    structureGgap = ''
    sp = structure.split()
    ssf = sp[0]
    ssf = ssf.replace('(', ')')
    for k in range(totalGgap):
        kk = k + 1
        sk = len(ssf) - kk + 1
        wk = 1 / (2 ** (totalGgap - kk))
        if kk == 1:
            for char11 in character:
                for char12 in character:
                    for char13 in character:
                        for char14 in character:
                            num1 = 0
                            for l1 in range(len(ssf) - kk - 3):
                                if ssf[l1] == char11 and ssf[l1 + 1] == char12 and ssf[l1 + kk + 2] == char13 and ssf[l1 + kk + 3] == char14:
                                    num1 = num1 + 1
                            f1 = wk * num1 / sk
                            string1 = str(f1) + separator
                            structureGgap = structureGgap + string1
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    for char23 in character:
                        for char24 in character:
                            num2 = 0
                            for l2 in range(len(ssf) - kk - 3):
                                if ssf[l2] == char21 and ssf[l2 + 1] == char22 and ssf[l2 + kk + 2] == char23 and ssf[l2 + kk + 3] == char24:
                                    num2 = num2 + 1
                            f2 = wk * num2 / sk
                            string2 = str(f2) + separator
                            structureGgap = structureGgap + string2
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        for char34 in character:
                            num3 = 0
                            for l3 in range(len(ssf) - kk - 3):
                                if ssf[l3] == char31 and ssf[l3 + 1] == char32 and ssf[l3 + kk + 2] == char33 and ssf[l3 + kk + 3] == char34:
                                    num3 = num3 + 1
                            f3 = wk * num3 / sk
                            string3 = str(f3) + separator
                            structureGgap = structureGgap + string3
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            num4 = 0
                            for l4 in range(len(ssf) - kk - 3):
                                if ssf[l4] == char41 and ssf[l4 + 1] == char42 and ssf[l4 + kk + 2] == char43 and ssf[l4 + kk + 3] == char44:
                                    num4 = num4 + 1
                            f4 = wk * num4 / sk
                            string4 = str(f4) + separator
                            structureGgap = structureGgap + string4
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            num5 = 0
                            for l5 in range(len(ssf) - kk - 3):
                                if ssf[l5] == char51 and ssf[l5 + 1] == char52 and ssf[l5 + kk + 2] == char53 and ssf[l5 + kk + 3] == char54:
                                    num5 = num5 + 1
                            f5 = wk * num5 / sk
                            string5 = str(f5) + separator
                            structureGgap = structureGgap + string5
    return structureGgap

def ComplexFeatureConstruction(feature1, feature2):
    fpair = ''
    if feature1 != '' and feature2 != '':
        f1, f2 = feature1.strip().split(' '), feature2.strip().split(' ')
        for i in range(len(f1)):
            a = float(f1[i])
            for j in range(len(f2)):
                b = float(f2[j])
                c = 1000 * (a + b) / 2
                fpair += str(c) + separator
    return fpair

def FusionComplexFeature(ListPair):
    TotalComplexFeature = []
    for LinePair in ListPair:
        miRNAname, lncRNAname, miRNAsequence, lncRNAsequence, miRNAstructure, lncRNAstructure, label = LinePair.split(',')
        # miRNA sequence k-mer
        SequencemiRNAkmer = SequencekmerExtract(miRNAsequence, Sequencekmertotal)
        # miRNA sequence g-gap
        SequencemiRNAGgap = SequenceGgapExtract2(miRNAsequence, SequenceGgaptotal)
        # miRNA structure k-mer
        StructuremiRNAkmer = StructurekmerExtract(miRNAstructure, Structurekmertotal)
        # miRNA structure g-gap
        StructuremiRNAGgap = StructureGgapExtract(miRNAstructure, StructureGgaptotal)
        # lncRNA sequence k-mer
        SequencelncRNAkmer = SequencekmerExtract(lncRNAsequence, Sequencekmertotal)
        # lncRNA sequence g-gap
        SequencelncRNAGgap = SequenceGgapExtract2(lncRNAsequence, SequenceGgaptotal)
        # lncRNA structure k-mer
        StructurelncRNAkmer = StructurekmerExtract(lncRNAstructure, Structurekmertotal)
        # lncRNA structure g-gap
        StructurelncRNAGgap = StructureGgapExtract(lncRNAstructure, StructureGgaptotal)
        # Complex Feature
        sequencekmer = ComplexFeatureConstruction(SequencemiRNAkmer, SequencelncRNAkmer)
        sequenceGgap = ComplexFeatureConstruction(SequencemiRNAGgap, SequencelncRNAGgap)
        structurekmer = ComplexFeatureConstruction(StructuremiRNAkmer, StructurelncRNAkmer)
        structureGgap = ComplexFeatureConstruction(StructuremiRNAGgap, StructurelncRNAGgap)
        # Fusion
        FeatureLine = sequencekmer + sequenceGgap + structurekmer + structureGgap + label
        TotalComplexFeature.append(FeatureLine)
    return TotalComplexFeature

def ConvertToMatrix(list, scale):
    X2, X3, X4, y = [], [], [], []
    NumberColumn = len(list[0].split(' '))
    for line in list:
        Sample = line.split(' ')
        Feature = Sample[0 : NumberColumn - 1]
        Feature2 = Feature
        while len(Feature2) % scale[0] != 0:
            Feature2.append('0.0')
        FeatureForm2 = np.array(Feature2).astype('float32').reshape(-1, scale[0])
        X2.append(FeatureForm2)
        Feature3 = Feature
        while len(Feature3) % scale[2] != 0:
            Feature3.append('0.0')
        FeatureForm3 = np.array(Feature3).astype('float32').reshape(-1, scale[2])
        X3.append(FeatureForm3)
        Feature4 = Feature
        while len(Feature4) % scale[4] != 0:
            Feature4.append('0.0')
        FeatureForm4 = np.array(Feature4).astype('float32').reshape(-1, scale[4])
        X4.append(FeatureForm4)
        Label = Sample[NumberColumn - 1].strip('\n')
        y.append(Label)
    X2 = np.array(X2).reshape(-1, scale[1], scale[0], 1)
    X3 = np.array(X3).reshape(-1, scale[3], scale[2], 1)
    X4 = np.array(X4).reshape(-1, scale[5], scale[4], 1)
    y = np.array(y).astype('int').reshape(-1, 1)
    y = np_utils.to_categorical(y, num_classes=2)
    return X2, X3, X4, y

def Kfold(X, Y, iteration, K):

    # separate the data
    totalpartX = len(X)
    partX = int(totalpartX / K)
    totalpartY = len(Y)
    partY = int(totalpartY / K)

    partXstart = iteration * partX
    partXend = partXstart + partX

    partYstart = iteration * partY
    partYend = partYstart + partY

    traindataP = np.array(X[0 : partXstart])
    traindataL = np.array(X[partXend : totalpartX])
    traindata = np.concatenate((traindataP, traindataL))
    testdata = np.array(X[partXstart : partXend])

    trainlabelP = np.array(Y[0 : partYstart])
    trainlabelL = np.array(Y[partYend : totalpartY])
    trainlabel = np.concatenate((trainlabelP, trainlabelL))
    testlabel = np.array(Y[partYstart : partYend])

    return traindata, trainlabel, testdata, testlabel

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

def Evaluation(y_test, resultslabel):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    AUC = roc_auc_score(y_test, resultslabel)
    for row1 in range(resultslabel.shape[0]):
        for column1 in range(resultslabel.shape[1]):
            if resultslabel[row1][column1] < 0.5:
                resultslabel[row1][column1] = 0
            else:
                resultslabel[row1][column1] = 1
    for row2 in range(y_test.shape[0]):
        if y_test[row2][0] == 0 and y_test[row2][1] == 1 and y_test[row2][0] == resultslabel[row2][0] and y_test[row2][1] == resultslabel[row2][1]:
            TP = TP + 1
        if y_test[row2][0] == 1 and y_test[row2][1] == 0 and y_test[row2][0] != resultslabel[row2][0] and y_test[row2][1] != resultslabel[row2][1]:
            FP = FP + 1
        if y_test[row2][0] == 1 and y_test[row2][1] == 0 and y_test[row2][0] == resultslabel[row2][0] and y_test[row2][1] == resultslabel[row2][1]:
            TN = TN + 1
        if y_test[row2][0] == 0 and y_test[row2][1] == 1 and y_test[row2][0] != resultslabel[row2][0] and y_test[row2][1] != resultslabel[row2][1]:
            FN = FN + 1
    if TP + FN != 0:
        SEN = TP / (TP + FN)
    else:
        SEN = 999999
    if TN + FP != 0:
        SPE = TN / (TN + FP)
    else:
        SPE = 999999
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 999999
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 999999
    return TP, FP, TN, FN, SEN, SPE, ACC, F1, AUC

def CNNLSTMAM(X_train, y_train, X_test, Frow, Fcolumn):
    # Model
    model = Sequential()
    # Convolution layer
    model.add(Convolution2D(batch_input_shape=(None, Fcolumn, Frow, 1), filters=32, kernel_size=Frow, activation='relu', strides=1, padding='same', data_format='channels_last'))
    # MaxPooling layer
    model.add(MaxPooling2D(pool_size=Frow, strides=Frow, padding='same', data_format='channels_last'))
    # Convolution layer
    model.add(Convolution2D(filters=64, kernel_size=Frow, activation='relu', strides=1, padding='same', data_format='channels_first'))
    # MaxPooling layer
    model.add(MaxPooling2D(pool_size=Frow, strides=Frow, padding='same', data_format='channels_last'))
    # Flatten layer
    model.add(TimeDistributed(Flatten()))
    # RNN layer
    model.add(LSTM(units=64, activation='tanh', return_sequences=True))
    # Attention mechanism
    model.add(SeqSelfAttention(attention_type='additive'))
    # model.add(SeqSelfAttention(attention_type='multiplicative'))
    # Flatten layer
    model.add(Flatten())
    # Fully-connected layer
    model.add(Dense(128, activation='relu'))
    # Dropout layer
    model.add(Dropout(0.5))
    # Fully-connected layer
    model.add(Dense(2, activation='softmax'))
    # optimizer
    adam = optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # training
    print('Training --------------')
    model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1)
    # test
    print('\nTesting---------------')
    # get the confidence probability
    score = model.predict(X_test)
    return score

ListData = np.load(PathData)
print('##################### Load data completed #####################\n')
ListFusionComplexFeature = FusionComplexFeature(ListData)
print('##################### Construct and fuse complex feature completed #####################\n')
X2, X3, X4, y = ConvertToMatrix(ListFusionComplexFeature, scale)
print('##################### Convert to matrix completed #####################\n')
TPsum, FPsum, TNsum, FNsum, SENsum, SPEsum, ACCsum, F1sum, AUCsum = [], [], [], [], [], [], [], [], []
for iteration in range(10):
    X_train2, y_train2, X_test2, y_test = Kfold(X2, y, iteration, 10)
    Score2 = CNNLSTMAM(X_train2, y_train2, X_test2, scale[0], scale[1])
    print('##################### Model 2-5930 completed #####################\n')
    X_train3, y_train3, X_test3, y_test = Kfold(X3, y, iteration, 10)
    Score3 = CNNLSTMAM(X_train3, y_train3, X_test3, scale[2], scale[3])
    print('##################### Model 3-3954 completed #####################\n')
    X_train4, y_train4, X_test4, y_test = Kfold(X4, y, iteration, 10)
    Score4 = CNNLSTMAM(X_train4, y_train4, X_test4, scale[4], scale[5])
    print('##################### Model 4-2965 completed #####################\n')
    ScoreE = GreedyFuzzyDecision(Score2, Score3, Score4)
    print('##################### Ensemble based on greedy fuzzy decision completed #####################\n')
    TP, FP, TN, FN, SEN, SPE, ACC, F1, AUC = Evaluation(y_test, ScoreE)
    print('##################### Evalucation completed #####################\n')
    print('####################################################################################\n')

    print('The ' + str(iteration + 1) + '-fold cross validation result')
    print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
    print('TPR:', TPR, 'TNR:', TNR, 'PPV:', PPV, 'NPV:', NPV, 'FNR:', FNR, 'FPR:', FPR, 'FDR:', FDR, 'FOR:', FOR)
    print('ACC:', ACC, 'F1:', F1, 'MCC:', MCC, 'BM:', BM, 'MK:', MK, 'AUC:', AUC)

    TPsum.append(TP)
    FPsum.append(FP)
    TNsum.append(TN)
    FNsum.append(FN)
    SENsum.append(TPR)
    SPEsum.append(TNR)
    ACCsum.append(ACC)
    F1sum.append(F1)
    AUCsum.append(AUC)

# print the average results
print('The average results')
print('\ntest mean TPR: ', np.mean(SENsum))
print('\ntest mean TNR: ', np.mean(SPEsum))
print('\ntest mean ACC: ', np.mean(ACCsum))
print('\ntest mean F1-score: ', np.mean(F1sum))
print('\ntest mean AUC: ', np.mean(AUCsum))

