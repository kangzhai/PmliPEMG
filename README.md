# PmliPEMG
The related data and scoure codes of PmliPEMG are privided by Q. Kang.

The latest version is updated on January 13, 2021.

# Introduction
PmliPEMG is a predictor for plant miRNA-lncRNA interaction. We enhance the information at multiple levels and build an ensemble deep learning model based on greedy fuzzy decision. PmliPEMG can be applied to the cross-species prediction and and shows better performance and stronger generalization ability than state-of-the-art predictors. It may also provide valuable references for related research.

# Dependency
Windows operating system

Python 3.6.5

Kreas 2.2.4

# Detail
BaseModel folder:

10 groups of trained base models mentioned in the paper.

Example folder:

The examples of input and the training-validation set mentioned in the paper.

Path.py:

Paths of input and output.

DataProcessing.py:

Function of data processing.

PredictionProcessing.py:

Function of prediction process.

SeqSelfAttention.py:

Code of attention mechanism proposed by W. Shi.

PmliPEMG (cross Validation).py:

Source code of cross validation in the paper.

PmliPEMG.py:

Code for predicting whether there has been interaction in unlabeled miRNA-lncRNA pairs.

# Usage
Open the console or powershell in the local folder and copy the following commands to run PmliPEMG. It is also feasible to run the codes using python IDE (such as pyCharm).

### PmliPEMG.py
Command: python PmliPEMG.py

Explanation:

It can predict whether there has been interaction in the unlabeled miRNA-lncRNA pairs. It can quickly predict large-scale interactions by  loading and integrating the trained base models. The users can adjust the path of input and output in "Path.py" to realize the prediction of local data. The input format must be consistent with that in the "Example" folder. This input format can be obtained directly through RNAfold in ViennaRNA package. The output is the predicted results, which lists miRNA name, lncRNA name and interaction/non-interaction. We will add more predicted information in future versions. To show the authenticity of the codes, we provide all 10 groups of base models mentioned in the paper. By default, 10 groups of base models independently predict the unlabeled samples and output the results. Due to the differences between the base models, the 10 groups of results will also vary. Users can comprehensively refer to these results, or manually adjust to use a group of base models for prediction. We will also integrate these results in future versions.

### PmliPEMG (Cross Validation).py
Command: python PmliPEMG(Cross Validation).py

Explanation:

It is the source code of cross validation mentioned in the paper, which can help the users repeat our experiment. It also shows the source code of the base model. Since the original data set was too large to upload, we compressed it as a ".zip" file. Before executing this code, the users need to unzip the "TrainingValidaitonSet.zip" in the "Example" folder to the current directory. This will be a relatively long process due to the need to retrain the base models. In addition to repeating the experiment, we hope that the code provides a valuable reference for users' research.

### Install-ViennaRNA-2.4.10_64bit.exe
When using "PmliPEMG.py" to the predict unlabeled samples, the input must be the correct format. The input format can be referred to "miRNA" and "lncRNA" in "Example" folder. This format can be obtained directly by using RNAfold (a RNA secondary structure extraction tool) in ViennaRNA package. Install-ViennaRNA-2.4.10_64bit.exe is the installation of ViennaRNA package that contains RNAfold. The latest version of ViennaRNA package can be also downloaded from https://www.tbi.univie.ac.at/RNA/. 

# Reference
The paper has been submitted. Please waiting for updating.
