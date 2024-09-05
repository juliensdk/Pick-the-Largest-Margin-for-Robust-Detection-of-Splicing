    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import scipy.stats as stats
import statistics
import os
import math 

#gather names of csv files containing latent margins for each training sample w.r.t. the different trained models
model_files = sorted(os.listdir("margins_data"))
model_files_list = []
for model_file in model_files:
    if 'model' in model_file:
        model_files_list.append(os.path.join("margins_data",model_file))

ref_model=[int(model.split('.')[0].split('_')[1]) for model in model_files if 'model' in model]
model_files_list=np.array(model_files_list)[np.argsort(ref_model)]

def calculate_statistics(column):
    Q1=column.quantile(0.25)
    Q2=column.quantile(0.5)
    Q3=column.quantile(0.75)
    I=Q3-Q1
    fence_min=column[column>Q1-1.5*I].min()
    fence_max=column[column<Q3+1.5*I].max()
    return [Q1,Q2,Q3,fence_min,fence_max]

def median_plot(distance,regret,distance_name,slide=0.1,gap=0.01,alpha=0.8,fontsize=10):
    min_d=round(distance.min(),2)
    max_d=round(distance.max(),2)
    N=[len(regret[np.logical_and(distance>i-slide,distance<i+slide)]) for i in np.arange(min_d,max_d,gap)]
    L1=[np.quantile(regret[np.logical_and(distance>i-slide,distance<i+slide)],0.25) for i in np.arange(min_d,max_d,gap)]
    L2=[np.median(regret[np.logical_and(distance>i-slide,distance<i+slide)]) for i in np.arange(min_d,max_d,gap)]
    L3=[np.quantile(regret[np.logical_and(distance>i-slide,distance<i+slide)],0.75) for i in np.arange(min_d,max_d,gap)]
    L4=[np.quantile(regret[np.logical_and(distance>i-slide,distance<i+slide)],0.9) for i in np.arange(min_d,max_d,gap)]

    
    plt.fill_between(np.arange(min_d,max_d,gap), L1,L3, color='grey', alpha=0.5)
    plt.plot(np.arange(min_d,max_d,gap),L4,c='black',linestyle='dotted',label='Q(90\%)',alpha=alpha)
    plt.plot(np.arange(min_d,max_d,gap),L3,c='black',linestyle='--',label='Q3',alpha=alpha)
    plt.plot(np.arange(min_d,max_d,gap),L2,label='median',c='black',alpha=alpha)
    plt.plot(np.arange(min_d,max_d,gap),L1,c='black',linestyle='--',label='Q1',alpha=alpha)
    
    #plt.xticks(list(np.arange(min_d-gap,max_d+gap,10*gap)),fontsize=fontsize)
    #plt.yticks(list(np.arange(int(regret.min()),int(regret.max()),5)),fontsize=fontsize)
    plt.grid()

    plt.xlabel(distance_name,fontsize=fontsize)
    plt.ylabel('Generalization gap (\%)',fontsize=fontsize)
    plt.legend(fontsize=fontsize//2+5)
    
    return N,np.arange(min_d,max_d,gap)
