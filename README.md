# EResCNN
Prediction of protein-protein interactions based on ensemble residual conventional neural network

###EResCNN uses the following dependencies:
* python 3.6 
* numpy
* scipy
* scikit-learn
* tensorflow

###Guiding principles:

**Feature extraction
  PAAC.m indicates pseudo amino acid composition
  MMI.m indicates multiple mutual information 
  PsePSSM.m indicates pseudo position-specific scoring matrix 
  ACC.m indicates auto covariance
  CTriad.py indicates Conjoint triad
  ebgw1.m ebgw2.m and ebgw3.m indicate encoding based on grouped weight

** Deep learning:
   EResCNN.py is the implementation of ensemble residual convolutional neural network.
   RCNN_test.py the implementation of residual convolutional neural network.
   Feature_evaluation_classifier.py is the implementation of conventional classifiers
