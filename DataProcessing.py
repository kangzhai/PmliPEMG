# Data process

import numpy as np
import math

separator, Sequencekmertotal, SequenceGgaptotal, Structurekmertotal, StructureGgaptotal = ' ', 3, 3, 3, 3

def PairConstruction(ListmiRNA, ListlncRNA):
    ListPair, NummiRNA, NumlncRNA = [], len(ListmiRNA), len(ListlncRNA)
    for IndmiRNA in range(NummiRNA):
        if '>' in ListmiRNA[IndmiRNA]:
            miRNAname = ListmiRNA[IndmiRNA].strip()
            miRNAsequence = ListmiRNA[IndmiRNA + 1].strip().replace('U', 'T')
            miRNAstructure = ListmiRNA[IndmiRNA + 2].strip()
            for IndlncRNA in range(NumlncRNA):
                if '>' in ListlncRNA[IndlncRNA]:
                    lncRNAname = ListlncRNA[IndlncRNA].strip()[1:]
                    lncRNAsequence = ListlncRNA[IndlncRNA + 1].strip().replace('U', 'T')
                    lncRNAstructure = ListlncRNA[IndlncRNA + 2].strip()
                    Pair = miRNAname + ',' + lncRNAname + ',' + miRNAsequence + ',' + lncRNAsequence + ',' + miRNAstructure + ',' + lncRNAstructure
                    ListPair.append(Pair)
    return ListPair

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
        miRNAname, lncRNAname, miRNAsequence, lncRNAsequence, miRNAstructure, lncRNAstructure = LinePair.split(',')
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
        FeatureLine = sequencekmer + sequenceGgap + structurekmer + structureGgap
        FeatureLine = FeatureLine.strip()
        TotalComplexFeature.append(FeatureLine)
    return TotalComplexFeature

def ConvertToMatrix(ListFusionComplexFeature, Frow):
    X_test = []
    Fcolumn = math.ceil((len(ListFusionComplexFeature[0].split(' '))) / Frow)
    for line in ListFusionComplexFeature:
        Feature = line.split(' ')
        while len(Feature) % Frow != 0:
            Feature.append('0.0')
        FeatureForm = np.array(Feature).astype('float32').reshape(-1, Frow)
        X_test.append(FeatureForm)
    X_test = np.array(X_test).reshape(-1, Fcolumn, Frow, 1)
    return X_test




