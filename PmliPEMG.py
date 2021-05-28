# PmliPEMG for predicting plant miRNA-lncRNA interaction

import numpy as np
import math
from Path import PathInput, PathOutput
from DataProcessing import PairConstruction, FusionComplexFeature, ConvertToMatrix
from PredictionProcessing import LoadModel, GreedyFuzzyDecision, AddLabel

# Load data
PathmiRNA, PathlncRNA = PathInput()
ListmiRNA = open(PathmiRNA, 'r').readlines()
ListlncRNA = open(PathlncRNA, 'r').readlines()
print('Data Loading Complete')
# Construct miRNA-lncRNA pair
ListPair = PairConstruction(ListmiRNA, ListlncRNA)
print('Pair Construction Complete')
# Extract features and construct complex feature
ListFusionComplexFeature = FusionComplexFeature(ListPair)
print('Feature Extraction and Fusion Complete')
# Convert complex feature vector to matrix
X_test2 = ConvertToMatrix(ListFusionComplexFeature, 2)
X_test3 = ConvertToMatrix(ListFusionComplexFeature, 3)
X_test4 = ConvertToMatrix(ListFusionComplexFeature, 4)
print('Matrix Conversion Complete')
# Prediction and ensemble
scale = ['2-5930', '3-3954', '4-2965']
Output = []
for it in range(10): # 10 independent predictions
    # Load base model
    Model2 = LoadModel(scale[0], it)
    Model3 = LoadModel(scale[1], it)
    Model4 = LoadModel(scale[2], it)
    # Prediction and get confidence probability
    Score2 = Model2.predict(X_test2)
    Score3 = Model3.predict(X_test3)
    Score4 = Model4.predict(X_test4)
    # Ensemble based on greedy fuzzy decision
    ScoreE = GreedyFuzzyDecision(Score2, Score3, Score4)
    # Add label
    Label = AddLabel(ScoreE, ListPair, it)
    Output += Label
print('The predicted results have been output!')
PathResult = PathOutput()
w = open(PathResult, 'w')
w.writelines(Output)
w.close()
