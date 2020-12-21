from __future__ import division
from __future__ import print_function
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import LeaveOneOut
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from math import exp
import numpy as np
import SIMLR
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import settings
from constructor import get_placeholder, get_model, get_model_2, format_data_new, get_optimizer, get_optimizer_2, update
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Encoder():
    def __init__(self, settings):
        self.iteration = settings['iterations']
        self.model = settings['model']
        
    def new_dataset_predicted_TV(self, emb, ztrTV, rearrangedTargetView, original_train_TV):
        simlr = SIMLR.SIMLR_LARGE(1, 10, 0)
        loo = LeaveOneOut()
        x = emb[:-1]
        for train, test in loo.split(x):
            test_index = test
            new_tr_tst_z_sv = np.vstack((x[train],x[test]))
            new_tr_z_tv = ztrTV[train]
            SALL, FALL,val, ind = simlr.fit(new_tr_tst_z_sv)
            SY, FY,val, ind = simlr.fit(new_tr_z_tv)
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
            
            # make prediction for each testing subject
            for testingSubject in range(1,2):
                # get this testing subject's rearranged index and original index
                tSubjectIndex = (new_tr_z_tv.shape[0]-2) + testingSubject
                tSubjectOriginalIndex = test_index
                # compute Tscore for each neighbor
                trustScore = np.ones((TS_bestNb,TS_bestNb))
                newWeight_TSW = np.ones(TS_bestNb)
                
                for neighbor in range(1,TS_bestNb+1):
                    neighborIndex = Index_ALL[tSubjectIndex,neighbor]
                    temp_counter = 0
                    while (neighborIndex  > new_tr_z_tv.shape[0]):
                    # best neighbor is a testing data 
                        temp_counter = temp_counter + 1
                        neighborIndex = Index_ALL[tSubjectIndex,(TS_bestNb + temp_counter)]

                    if (temp_counter != 0):
                        neighborSequence = TS_bestNb + temp_counter
                    else:
                        neighborSequence = neighbor

                        print("---",neighborIndex)
                        # get top nb neighbors in mappedX
                        neighborListX = Index_ALL[neighborIndex,1:TS_bestNb+1]
                        # get top nb neighbors in mappedY
                        neighborListY = Index_Y[neighborIndex,1:TS_bestNb+1]
                        # calculate trust score
                        trustScore[TS_bestNb-1,neighbor-1] = len(np.intersect1d(np.array(neighborListX),np.array(neighborListY)))
                        # calculate new weight (TS * Similarity)
                        newWeight_TSW[neighbor-1] = exp(trustScore[TS_bestNb-1,neighbor-1] / TS_bestNb * Bvalue_ALL[tSubjectIndex,neighborSequence])
                      
                        
                # reconstruct with Tscore and similarity weight
                innerPredict_TSW = np.zeros(original_train_TV.shape[1])[np.newaxis]
                # summing up the best neighbors
                for j1 in range(0,TS_bestNb):
                    tr = (rearrangedTargetView[:,Index_ALL[tSubjectIndex,j1]])[np.newaxis]
                    if j1 == 0:
                        innerPredict_TSW = innerPredict_TSW.T + tr.T * newWeight_TSW[j1]
                    else:
                        innerPredict_TSW = innerPredict_TSW + tr.T * newWeight_TSW[j1]
                        
                   
                # scale weight to 1
                Scale_TSW = sum(newWeight_TSW)
                innerPredict_TSW = np.divide(innerPredict_TSW, Scale_TSW)
                if(test==0):
                    all_predictedTV_tr = np.c_[innerPredict_TSW]
                    #print("=====>>>> First subject in the Hidden SIMLR")
                else:
                    all_predictedTV_tr = np.c_[all_predictedTV_tr,innerPredict_TSW]
          
        #print("=====>>>> Last subject in the Hidden SIMLR")
        resul = (all_predictedTV_tr)[np.newaxis]
        outputs = all_predictedTV_tr.T
        return outputs
        
      
    def erun(self, adj, features, hiddenSIMLR, original_train_TV, ztrTV, rearrangedTargetView):
        tf.reset_default_graph()
        
        model_str = self.model
        
        # formatted data
        feas = format_data_new(adj, coo_matrix(features))
        
        # Define placeholders
        placeholders = get_placeholder(feas['adj'])
       
        # construct model
        d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])
        
        # Optimizer
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

        if(hiddenSIMLR == "No_hidden_SIMLR"):
           
            # Initialize session
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
           
            # Train model
            for epoch in range(self.iteration):
                emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], features, hiddenSIMLR, 1)
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
                if (epoch+1) == 5:
                    break

        else:#(hiddenSIMLR == "Yes_hidden_SIMLR"):
            d_real_TV, discriminator2, ae_model2 = get_model_2(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])
            #print(tf.trainable_variables())
            # Optimizer
            opt2 = get_optimizer_2(model_str, ae_model, discriminator, discriminator2, placeholders, feas['pos_weight'], feas['norm'], d_real_TV, feas['num_nodes'])
            
            # Initialize session
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
           
            # Train model
            for epoch in range(self.iteration):
                if (epoch+1) == 5:
                        break
                if(epoch%2==0):#even
                    emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], features, "No_hidden_SIMLR", 1)
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
                    d_f = self.new_dataset_predicted_TV(emb, ztrTV, rearrangedTargetView, original_train_TV)
                    
                else:#odd
                    emb, avg_cost = update(ae_model, opt2, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], original_train_TV, hiddenSIMLR, d_f)
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
                    
        
        return emb
