#!/usr/bin/python
# -*- coding: utf8 -*-

    """
    Main function of Learning-guided Graph Dual Adversarial Domain Alignment (LG-DADA)framework
    for predicting a target brain graph from a source brain graph.
    
    The original paper can be found in: https://www.sciencedirect.com/science/article/pii/S1361841520302668
    Alaa Bessadok, Mohamed Ali Mahjoub, and Islem Rekik. "Brain graph synthesis by dual adversarial domain alignment
    and target graph prediction from a source graph", Medical Image Analysis Journal 2020.
    ---------------------------------------------------------------------
    
    This file contains the implementation of four key steps of our LG-DADA framework:
    (1) feature extraction and clustering, (2) adversarial domain alignment, (3) dual adversarial regularization and (4) target graph prediction.
        main(sourceGraph,targetGraph,labels,settings)
                Inputs:
                        sourceGraph: (n × m) matrix stacking the source graphs of all subjects
                                     n the total number of subjects
                                     m the number of features
                        targetGraph: (n × m) matrix stacking the target graphs of all subjects
                                     n the total number of subjects
                                     m the number of features
                        settings:    store the neural network settings such as the dimension of the embedded graphs
                                     and the type of autoencoder we choosed (variational or simple autoencoder)
                Output:
                        predicted_target_graphs: (n × m) matrix stacking the the predicted target graphs
                        
    
    To evaluate our framework we used Leave-One-Out crossvalidation strategy.
        
     Sample use:
     dataset_predicted_target = main(sourceGraph,targetGraph,settings)
    
    ---------------------------------------------------------------------
    Copyright 2020 Alaa Bessadok, Sousse University.
    Please cite the above paper if you use this code.
    All rights reserved.
    """

import tensorflow as tf
from scipy import stats
from math import exp
import numpy as np
import SIMLR
import time
import settings
from encoder import Encoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error


start = time.time()

def main(sourceGraph,targetGraph,settings):
    # initialisation
    subject = 150
    overallResult_PCC = np.zeros((subject,32))
    overallResult_TSW = np.zeros((subject,32))
    predictedTargetGraph = np.empty((0,targetGraph.shape[1]), int)

    ## STEP 1: feature extraction and clustering
    print("======SIMLR for Clustering======")
    c = 2
    simlr = SIMLR.SIMLR_LARGE(c, 50, 0)
    S, F, val, ind = simlr.fit(sourceGraph)
    y_pred = simlr.fast_minibatch_kmeans(F,c)
    
    y_pred = y_pred.tolist()
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    # Split the data into training and testing sets
    loo = LeaveOneOut()
    loo.get_n_splits(sourceGraph)

    for train_index, test_index in loo.split(sourceGraph):
        rearrangedPredictorView = np.concatenate((np.transpose(sourceGraph[train_index]), np.transpose(sourceGraph[test_index])),axis = 1)
        rearrangedTargetView = np.concatenate((np.transpose(targetGraph[train_index]),np.transpose(targetGraph[test_index])),axis = 1)
        
        label_of_test_index = y_pred[test_index[0]]
        train_index = get_indexes(label_of_test_index,y_pred)
        
        print("Testing subject number: ", test_index[0]," of cluster ", label_of_test_index)
        print("Training subjects: ", len(train_index)," of cluster ", label_of_test_index)
        
    ## STEP 2: Domain Alignment for training samples
        simlr1 = SIMLR.SIMLR_LARGE(1, 10, 0)
        enc = Encoder(settings)
    
        train__TV_A = targetGraph[train_index]
        SV = sourceGraph[train_index]
        print("Encode the target graph...")
        Simlarity2, _,_, _ = simlr1.fit(train__TV_A)
        encode_train__TV_A = enc.erun(Simlarity2, sourceGraph[train_index],"No_hidden_SIMLR",1,1,1)
        
    ## STEP 3: Dual Adversarial Regularization of source graph embedding for training and testing samples
        test__train__SV = np.vstack((sourceGraph[train_index],sourceGraph[test_index]))
        print("Encode the source view of the TRAIN subjects and the TEST subject...")
        Simlarity1, _,_, _ = simlr1.fit(test__train__SV)
        encode_test__train__SV = enc.erun(Simlarity1, test__train__SV,"Yes_hidden_SIMLR", train__TV_A, encode_train__TV_A, rearrangedTargetView)
      
    ## Connectomic Manifold Learning using SIMLR
        SALL, FALL,val, ind = simlr1.fit(encode_test__train__SV)
        SY, FY,val, ind = simlr1.fit(encode_train__TV_A)
        # number of neighbors for trust score
        TS_bestNb = 5
        # get best TS_benstNb neighbors for everyone
        sall = SALL.todense()
        Index_ALL = np.argsort(-sall, axis=0)
        des = np.sort(-sall, axis=0)
        Bvalue_ALL = -des

        sy = SY.todense()
        Index_Y = np.argsort(-sy, axis=0)
        desy = np.sort(-sy,axis=0)
        Bvalue_Y = -desy

        ## STEP 4: Target Graph Prediction
        # make prediction for each testing subject
        for testingSubject in range(1,2):
            print "testing subject:", test_index[0]
            # get this testing subject's rearranged index and original index
            tSubjectIndex = (SV.shape[0]-2) + testingSubject
            tSubjectOriginalIndex = test_index
            # compute Tscore for each neighbor
            trustScore = np.ones((TS_bestNb,TS_bestNb))
            newWeight_TSW = np.ones(TS_bestNb)

            for neighbor in range(0,TS_bestNb):
                neighborIndex = Index_ALL[tSubjectIndex,neighbor]
                temp_counter = 0
                while (neighborIndex  > SV.shape[0]):
                # best neighbor is a testing data 
                    temp_counter = temp_counter + 1
                    neighborIndex = Index_ALL[tSubjectIndex,(TS_bestNb + temp_counter)]

                if (temp_counter != 0):
                    neighborSequence = TS_bestNb + temp_counter
                else:
                    neighborSequence = neighbor

                    print('----',neighborIndex)
                    if (neighborIndex == Index_Y.shape[0]):
                        continue
                    # get top nb neighbors in mappedX
                    neighborListX = Index_ALL[neighborIndex,0:TS_bestNb]
                    # get top nb neighbors in mappedY
                    neighborListY = Index_Y[neighborIndex,0:TS_bestNb]
                    # calculate trust score
                    trustScore[TS_bestNb-1,neighbor] = len(np.intersect1d(np.array(neighborListX),np.array(neighborListY)))
                    # calculate new weight (TS * Similarity)
                    newWeight_TSW[neighbor] = exp(trustScore[TS_bestNb-1,neighbor] / TS_bestNb * Bvalue_ALL[tSubjectIndex,neighborSequence])
                    
            #reconstruct with Tscore and similarity weight
            innerPredict_TSW = np.zeros(SV.shape[1])[np.newaxis]
            #summing up the best neighbors
            for j1 in range(0,TS_bestNb):
                tr = (rearrangedTargetView[:,Index_ALL[tSubjectIndex,j1]])[np.newaxis]
                if j1 == 0:
                    innerPredict_TSW = innerPredict_TSW.T + tr.T * newWeight_TSW[j1]
                else:
                    innerPredict_TSW = innerPredict_TSW + tr.T * newWeight_TSW[j1]

            # scale weight to 1
            Scale_TSW = sum(newWeight_TSW)
            innerPredict_TSW = np.divide(innerPredict_TSW, Scale_TSW)
            
            # calculate PCC and MAE
            tr2 = (rearrangedTargetView[:,tSubjectIndex])[np.newaxis]
            resulttsw =abs(tr2.T - innerPredict_TSW)
            iMAE_TSW = mean_absolute_error(tr2.T, innerPredict_TSW)
            overallResult_TSW[tSubjectOriginalIndex,TS_bestNb] = overallResult_TSW[tSubjectOriginalIndex,TS_bestNb] + iMAE_TSW
            [r,p] = stats.pearsonr(tr2.T, innerPredict_TSW)
            overallResult_PCC[tSubjectOriginalIndex,TS_bestNb] = overallResult_PCC[tSubjectOriginalIndex,TS_bestNb] + r
            
            predictedTargetGraph = np.append(predictedTargetGraph, innerPredict_TSW.T, axis=0)

            print test_index[0]
            
            
    pcc = np.mean(overallResult_PCC,axis=0)
    print("Pearson Correlation Coefficient: ", pcc)
    mae = np.mean(overallResult_TSW,axis=0)
    print("Mean Absolute Error: ", mae)
     
    
    return predictedTargetGraph

           
## Simulate graph data for simply running the code
## in this exemple, the source and target matrices have different statistical distributions
mu, sigma = 0.2226636809, 0.02720207221 # mean and standard deviation
sourceGraph = np.random.normal(mu, sigma, (150,595))
mu, sigma = 0.0830806568, 0.01338490182
targetGraph = np.random.normal(mu, sigma, (150,595))

model = 'arga_ae' #autoencoder/variational autoencoder
settings = settings.get_settings_new(model)
predicted_target_graphs = main(sourceGraph, targetGraph, settings)


end = time.time()
print(end - start)

# -*- coding: utf-8 -*-