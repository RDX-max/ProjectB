#!/usr/bin/python
print('123')
from __future__ import division
from math import sqrt
import numpy as np
from numpy import linspace
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import pylab
from sklearn.metrics import auc

from sklearn.metrics import average_precision_score

predicted_scores=[]
pure_all_scores=[]
pure_predicted_scores=[]
filename=open('total.txt','r')
for pred in filename:
     pred=pred.split()
     #print pred
     x4=[float(b2) for b2 in pred]
     x5=sum(x4)/len(x4)
     predicted_scores.append(x5)
    
all_lines=[line.rstrip() for line in open('total_casp10.txt')]

#predicted_scores=np.array([predicted_scores])
for x,y in zip(all_lines,predicted_scores):
      if x=='D':
          pure_all_scores.append(0)
          pure_predicted_scores.append(y)
      if x=='O':
          pure_all_scores.append(1)
          pure_predicted_scores.append(y)
      
pure_all_scores=np.array([pure_all_scores])
pure_predicted_scores=np.array([pure_predicted_scores])         
fpr, tpr, thresholds = metrics.roc_curve(pure_all_scores.T, pure_predicted_scores.T, pos_label=0)
#print roc_auc_score(pure_all_scores.T, pure_predicted_scores.T)
precision, recall, thresholds= precision_recall_curve(pure_all_scores.T, pure_predicted_scores.T)
#print thresholds[0]
#pylab.plot(recall, precision) 
#pylab.show()
#auc = auc(recall, precision)
#print  auc
#print average_precision_score(pure_all_scores.T, pure_predicted_scores.T)




zlim=0.5
def calcMCC(probs,Z,doplot=True):
   plims=linspace(-5.0,16.1,50)#change the last number to get more fine-grain number...
   allprecs=[];allrecs=[]
   for plim in plims:
    zlow=Z>zlim#supposed to be disordered
    zhig=Z<=zlim
    positives=probs[zlow]#the probs corresponding to disordered resideus
    negatives=probs[zhig]
    nP=len(positives)
    TP=sum(positives>plim)#the probs > 0.5 (or specific thr)
    FN=nP-TP
    nN=len(negatives)
    #print nN
    TN=sum(negatives<=plim)
    FP=nN-TN
    MCC=(TP * TN - FP * FN)/sqrt(nP * nN * (FP + TN) * (TP + FN))
    TPR=TP*1.0/nP; 
    FNR=1-TPR; 
    TNR=TN*1.0/nN; 
    FPR=1-TNR

    tot=nP+nN+0.0
    #print 'MCC',zlim,MCC,nP,nN
    prec=TP*1.0/(TP+FP)
    accu=0.5*(TP*1.0/(TP+FN)+TN*1.0/(TN+FP))
    #print 'details:  %6.4f %6.4f %6.4f %6.4f  %6.4f %6.4f %s'%(plim,TPR,TNR,FPR,FNR,prec,accu)
    if TP+FP>0 and nP>0:
      allprecs.append(prec)
      allrecs.append(TPR)
   AUC_PR=sum(allprecs)/len(allprecs)
   print 'AUC_PR %6.4f '%(AUC_PR)
   if doplot:pylab.plot(allrecs,allprecs)#and youll need also pylab.show() later...
   return AUC_PR
############### convert to probabilities ?//

zpred=pure_predicted_scores.T
s01=pure_all_scores.T
order=s01<0.5
dis=s01>0.5
zdis=zpred[dis]
zord=zpred[order]
#print len(zdis),len(zord)
#print np.average(zdis),np.average(zord)
pylab.hist(zdis,fc='r',alpha=0.5,bins=np.arange(-5,16.5,0.5))
pylab.hist(zord,fc='b',alpha=0.5,bins=np.arange(-5,16.5,0.5))
pylab.show()



calcMCC( pure_predicted_scores.T,pure_all_scores.T,doplot=True)
pylab.show()
