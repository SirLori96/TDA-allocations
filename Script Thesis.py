# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:23:05 2021

@author: 111949
"""

#yahoo finance package to download prices
import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import datetime

#packages for optimization and clustering
from scipy.optimize import minimize 
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster,cophenet
#from scipy.spatial.distance import dist
from scipy.spatial import distance
 
import seaborn as sns

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
import copy
import os

###############################################################################
def getIVP(cov,**kargs):
# Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp


def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar




def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()


def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[int(j):int(k)] for i in cItems for j,k in ((0,len(i)/2), \
                                                   (len(i)/2,len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w

def getRecBipart2(cov,sortIx,clusters_list):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=clusters_list # initialize all items in one cluster
    
    #calcolo le varianze dei portafogli
    cItems_var_list=[]
    for i in range(0,len(cItems)):
        cItems0=cItems[i] # cluster 1
        cVar0=getClusterVar(cov,cItems0)
        cItems_var_list.append(cVar0)
    
    #calcolo quando dare al singolo portafoglio
    alpha_list=[]
    for i in range(0,len(cItems_var_list)):
        alpha=1-cItems_var_list[i]/sum(cItems_var_list)
        alpha_list.append(alpha)
    #modifico per il numero di elementi
    alpha_list_corrected=[]
    for i in range(len(clusters_list)):
        alpha_list_corrected.append(alpha_list[i]*len(clusters_list[i])/len(w))
    #correggo i pesi in modo tale che la somma faccia 1
    alpha_sum1 = pd.Series(alpha_list_corrected)/sum(alpha_list_corrected)
    
    for z in range(len(clusters_list)):
        cItems_=[clusters_list[z]]
        w_clust= w[cItems_[0]]
        while len(cItems_)>0:
            cItems_=[i[int(j):int(k)] for i in cItems_ for j,k in ((0,len(i)/2), \
                                                       (len(i)/2,len(i))) if len(i)>1] # bi-section
            for i in range(0,len(cItems_),2): # parse in pairs
                cItems0=cItems_[i] # cluster 1
                cItems1=cItems_[i+1] # cluster 2
                cVar0=getClusterVar(cov,cItems0)
                cVar1=getClusterVar(cov,cItems1)
                alpha=1-cVar0/(cVar0+cVar1)
                w_clust[cItems0]*=alpha # weight 1
                w_clust[cItems1]*=1-alpha # weight 2
        #trovo la somma er cui dividere in modo tale che la somma dei pesi 
        #risultanti dia esattamente il peso del cluster ottenuto in alpha_sum1
        w_clust=w_clust*alpha_sum1[z]/sum(w_clust)
        w[clusters_list[z]]=w_clust
    return w


def sharpe_ratio(return_series):
    mean = return_series.mean() 
    sigma = return_series.std() 
    return (mean / sigma) * (255 **0.5)

def risk_adv_coef(return_series):
    mean = return_series.mean() 
    var = return_series.var() 
    return (mean / var) 


#mapper is a dictionary in which the columns' names of the prices df and the columns' names
#of the text-signals-df are related. This is indespensable so that the 2 dfs' columns
#are sorted in the same way. In this dict the key must be the text-signal-column name, 
#and the value must be the prices df column name.
def from_signal_to_w(mapper, signal_df, perc):
    df_copy = signal_df.copy()
    count_df = df_copy.abs().sum(1)
    weight_list=[]
    for row in range(len(df_copy)):
        row_list = df_copy.iloc[row,:].T
        if count_df.iloc[row] != 0:
            weight_row = perc/count_df.iloc[row]
            row_list[row_list != 0] = weight_row
            weight_eq_we = (1-perc)/(df_copy.shape[1] - count_df.iloc[row])
            row_list[row_list == 0] = weight_eq_we
        elif count_df.iloc[row] == 0:
            weight_eq_we = 1/df_copy.shape[1]
            row_list[row_list == 0] = weight_eq_we
        weight_list.append(pd.DataFrame(row_list.values).T)
    weight_df = pd.concat(weight_list, 0)
    weight_df.index = df_copy.index
    weight_df.columns = df_copy.columns
    sort_col = []
    for col in weight_df.columns:
        sort_col.append(mapper[col])
    weight_df.columns = sort_col
    return weight_df

#this function operates in the same way as the previous but without sorting the columns.
#To use this function, the dfs must be sorted before.
def from_signal_to_w_unsort(signal_df, perc):
    df_copy = signal_df.copy()
    count_df = df_copy.abs().sum(1)
    weight_list=[]
    for row in range(len(df_copy)):
        row_list = df_copy.iloc[row,:].T
        if count_df.iloc[row] != 0:
            weight_row = perc/count_df.iloc[row]
            row_list[row_list != 0] = weight_row
            weight_eq_we = (1-perc)/(df_copy.shape[1] - count_df.iloc[row])
            row_list[row_list == 0] = weight_eq_we
        elif count_df.iloc[row] == 0:
            weight_eq_we = 1/df_copy.shape[1]
            row_list[row_list == 0] = weight_eq_we
        weight_list.append(pd.DataFrame(row_list.values).T)
    weight_df = pd.concat(weight_list, 0)
    weight_df.index = df_copy.index
    weight_df.columns = df_copy.columns
    return weight_df




def cm_to_inch(value):
    return value/2.54    

def ann_ret(ce, years):
    return ((ce.iloc[-1] / ce.iloc[0])**(1/years) -1) 
###############################################################################

#Downloading prices

#starting and end dates (arbitrarly chosen)
start= pd.to_datetime('2010-01-01')
end= pd.to_datetime('2021-07-30')

corn= yf.download('ZC=F',start, end)

wheat= yf.download('ZW=F',start, end)

soybeans= yf.download('ZS=F',start, end)


#S&P tech 
apple= yf.download('AAPL',start, end)

intel= yf.download('INTC',start, end)

microsoft= yf.download('MSFT',start, end)

amazon=yf.download('AMZN', start, end)

#S&P healthcare
pfizer= yf.download('PFE',start, end)

#europe
bmw= yf.download('BMW.DE',start, end)

allianz=yf.download('ALV.DE',start, end)

enel= yf.download('ENEL.MI',start,end)

danone=yf.download('BN.PA',start, end)

#india
ind_auto= yf.download('BAJAJ-AUTO.NS',start, end)

ind_unilever= yf.download('HINDUNILVR.NS',start, end)

ind_telecom= yf.download('BHARTIARTL.NS',start, end)


#defining the assets' names
comp_list_names=['corn','wheat','soybeans','apple','intel','microsoft','pfizer','danone',
           'amazon','bmw','allianz','enel',
           'ind_auto','ind_unilever','ind_telecom']

#defining the list with all prices. This list must respect the order of the previous one
comp_list_prices=[corn,wheat,soybeans,apple,intel,microsoft,pfizer,danone,
           amazon,bmw,allianz,enel,ind_auto,ind_unilever,ind_telecom]


#in this function you have to input the 2 lists in the same form of the above ones.
#One must have companies name in a string format, the other must have prices.
def create_prices_df_from100(comp_list_names, comp_list_prices):
    #Build the dictionary with assets' prices inside. The prices considered are only
    #the "Open" prices, that are prices at the moment of the market opening.
    hloc_dict={}
    for comp,obj in zip(comp_list_names,comp_list_prices):
        hloc_dict[comp]=obj.Open
        hloc_dict[comp].name=comp
        
    #appending all prices in a list and then building a dataframe with all prices.   
    prices_list=[]
    for key in hloc_dict.keys():
        prices_list.append(hloc_dict[key]) 
    
    #create prices df
    prices_df=pd.concat(prices_list,1)
    
    #forward fill e drop na 
    prices_df.fillna(method='ffill',inplace=True)
    prices_df.dropna(inplace=True)
    
    # rescaling all prices such that they all start from 100
    prices_df_from100=100*prices_df/prices_df.iloc[0,:]
    
    return prices_df_from100





# =============================================================================
# pure HRP in expanding window
# =============================================================================
        
#Splitting the prices_df_from100 into 2 parts: the first one will contain
#the first 5 years (until 2014-12-31), the second one the remaining prices.
# The HRP will be run on the first 5 years to have a sufficient amount of data
#to estimate the variance-covariance matrix and to do the clustering correctly.

def splitting_prices_df(prices_df_from100):
    #defining the first date of the prices horizon
    start_df=prices_df_from100.index[0]
    #defining the end of the 5-years period
    end_2014=pd.to_datetime('2014-12-31')
    #getting the first 5 years
    prices_df_ante2015=prices_df_from100.loc[start_df:end_2014,:]
    
    #getting the remaining prices and rescaling them such that they start from 100
    prices_df_from2015 = prices_df_from100.drop(prices_df_ante2015.index)
    prices_df_from2015 = 100*prices_df_from2015/prices_df_from2015.iloc[0,:]
    
    return prices_df_ante2015,prices_df_from2015



#this function gets you the dates of every trimester.
def get_trimester_dates(prices_df_from2015):
    #from this point i get a list containing the beginning date of every trimester
    quarter_dates=pd.Series(prices_df_from2015.index).dt.to_period('Q').dt.to_timestamp()
    trimesters = []
    for uniq_trim in quarter_dates:
        if uniq_trim not in trimesters:
            trimesters.append(uniq_trim)
        else:
            continue
    
    #since there might be some months in which the trimester doesn t coincide with
    #the exact first of the month, this function allow to get exactly the begginning 
    #of the trimester according to the prices dataframe
    true_quarters_date=[]
    for elem in trimesters:
        res = min(prices_df_from2015.index, key=lambda sub: abs(sub - pd.to_datetime(elem)))
        true_quarters_date.append(res)
    
    return true_quarters_date


#compute correlation matrix
def get_corr_matr(prices_df_from100, start):
    prices_df_subset=prices_df_from100.loc[:start,:]

    all_ret=prices_df_subset.pct_change(1).dropna()
    #getting the correlation matrix
    corr_mat=all_ret.corr()
    
    return corr_mat


#compute the distances between correlation matrix values
def compute_distances(corr_mat):
    #initialize an empty df
    dist_df=pd.DataFrame(data=None, index=corr_mat.index , columns=corr_mat.columns)
    
    for col in range(0,corr_mat.shape[1]):
        for f in range(0,corr_mat.shape[0]):
            if f==col:
                dist_df.iloc[f,col]=0
            else:
                #distances between the row "i" and column "col"
                D=(0.5*(1-corr_mat.iloc[f,col]))**(1/2)
                dist_df.iloc[f,col]=D
    return dist_df


#compute the euclid dist.
def compute_euclid_dist(dist_df):
        #calcolo le distanze euclidee
    euclid_df= pd.DataFrame(data=None, index=dist_df.index , columns=dist_df.columns)
    
    for col in range(0,dist_df.shape[1]):
        euclid_df.iloc[col,col]=0
        a=dist_df.iloc[:,col]
        col_list=list(range(0,dist_df.shape[0]))
        col_list.remove(col)
        for col2 in col_list:
            b=dist_df.iloc[:,col2]
            #calcolo distanze euclidee trai vari vettori
            D=distance.euclidean(a, b)
            euclid_df.iloc[col,col2]=D
    return euclid_df



#dictionary where dove numbers from 0 to 15 are the relative assets
def name_dict_for_clust(euclid_df):
    name_dict={}
    for i,col in enumerate(euclid_df.columns):
        name_dict[i]=col
    return name_dict

#function that is needed to transform the numbers that correspond to companies' names.
def leaf_label_func(name_dict,id):
    return name_dict[id]

#this function is needed to check the best number of cluster by looking at dendrograms
def clustering(euclid_df, plot=False):
    #controllo quanti clusters mi consiglia il metodo classico
    Z=linkage(euclid_df, method='complete', optimal_ordering=False)
    
    if plot == True:
        dendrogram(Z, p=15, leaf_rotation=90., leaf_font_size=10., leaf_label_func=leaf_label_func)
        plt.title('HIERARCHICAL CLUSTERING DENDROGRAM')
        plt.xlabel('Cluster size')
        plt.ylabel('Distance')
        plt.axhline(y=500)
        plt.axhline(y=150)
        plt.show()
    
    return Z
    


def quasi_diagonalization(prices_df_from100, Z):
    prices_df_subset=prices_df_from100.loc[:start,:]
    
    all_ret=prices_df_subset.pct_change(1).dropna()
    
    #genero la matrice varianza covarianza
    var_cov_mat=all_ret.cov()
    
    sorted_elem=getQuasiDiag(Z)
    
    sorted_cov_mat=pd.DataFrame(index=var_cov_mat.index)
    for elem in sorted_elem:
        sorted_cov_mat=pd.concat([sorted_cov_mat,var_cov_mat.iloc[:,elem]],1)
        
    #sorting rows as well
    sorted_cov_mat=sorted_cov_mat.reindex(sorted_cov_mat.columns)
    
    return var_cov_mat, sorted_cov_mat



def recursive_bisection(var_cov_mat, sorted_cov_mat):
    #getting allocations for a trimester
    allocations=getRecBipart(var_cov_mat, sorted_cov_mat.columns)
    allocations=pd.DataFrame(allocations)
    allocations=allocations.reindex(comp_list_names).T
    
    return allocations


def get_equity_curve(trimester_prices, allocations):
    
    trimester_ret = trimester_prices.pct_change(1).dropna()
    
    weight_hrp = pd.DataFrame(columns=allocations.columns)
    z=0
    while z <len(trimester_ret):
        weight_hrp=weight_hrp.append(allocations)
        z+=1
        
    weight_hrp.index=trimester_ret.index
    ret_hrp=weight_hrp*trimester_ret
        
    ret_hrp= ret_hrp.sum(1)
    
    rend_cum_hrp = np.hstack([100, ret_hrp + 1])
    
    ce_hrp = np.cumprod(rend_cum_hrp)
    ce_hrp=pd.Series(ce_hrp).iloc[:-1]
    ce_hrp.index=trimester_ret.index        
    return ce_hrp, rend_cum_hrp


   






# =============================================================================
# MARKOWITZ'S CLA
# =============================================================================
        

     
def get_cov_and_ret(start, prices_df_from100):
    prices_df_subset=prices_df_from100.loc[:start,:]
            
   #compute returns 
    returns = prices_df_subset/prices_df_subset.shift(1).dropna()
    returns.dropna(inplace= True)
    
    #compute mean returns
    mean_ret= returns.mean()
    #compute var-cov matrix
    Sigma= returns.cov()
    return mean_ret, Sigma


def compute_minimization(start, prices_df_from100):
    mean_ret, Sigma = get_cov_and_ret(start, prices_df_from100)
    
    #inner functions to make the minimization work
    def negativeSR(w):
        w=np.array(w)
        R=np.sum(mean_ret)
        V=np.sqrt(np.dot(w.T,np.dot(Sigma,w)))
        SR=R/V
        return -1*SR
    
    
    def checkSumToOne(w):
        return np.sum(w)-1
    #maximizing the sharpe & minimizing the volatility 
    #initialize the "equally weighted vector". The length is the same of the 
    #number of portfolio assets
    w0=[1/len(comp_list_names)] * len(comp_list_names)
    #define boundaries
    bounds=[(0,1)] * int(len(comp_list_names))
    
    #define inner functions to make the minimization work
    
    
    #defining minimization settings
    constraints=({'type':'eq','fun':checkSumToOne})
    #running minimization
    w_opt=minimize(negativeSR,w0,method='SLSQP', bounds=bounds,constraints=constraints)
    
    w_opt_mrkwz = pd.DataFrame(w_opt.x)
    w_opt_mrkwz.index = Sigma.columns
    w_opt_mrkwz = w_opt_mrkwz.T
    return w_opt_mrkwz




def investing_next_trim(start, end, prices_df_from100,w_opt_mrkwz):
    #creo la curva equity per il trimestre successivo
    trimester_prices = prices_df_from100.loc[start:end,:]
    
    trimester_ret = trimester_prices.pct_change(1).dropna()
    
    #creo un df con tanti pesi (tutti uguali) quanti i giorni all'interno del trimestre
    weigh_mrkwz = pd.DataFrame(columns=prices_df_from100.columns)
    z=0
    while z <len(trimester_ret):
        weigh_mrkwz=weigh_mrkwz.append(w_opt_mrkwz, ignore_index=True)
        z+=1
        
    weigh_mrkwz.index=trimester_ret.index
    
    ret_mrkwz=weigh_mrkwz*trimester_ret
    
    ret_mrkwz= ret_mrkwz.sum(1)
    
    
    rend_cum_mrkwz = np.hstack([100, ret_mrkwz + 1])
    
    return rend_cum_mrkwz

    
    
#CLA MAIN
def compute_CLA(prices_df_from100):
    #preparing the prices for the expanding window training.
    prices_df_pre2015,prices_df_from2015=splitting_prices_df(prices_df_from100)
    true_quarters_date = get_trimester_dates(prices_df_from2015)


    ce_mrkwz_tot=pd.Series()
    ce_mrkwz_tot.name = 'ce_mrkwz'

    #price dates until 2021-06-30
    index = prices_df_from2015.index[:-21]



    for i in range(len(true_quarters_date)):
        #condizione per far terminare il loop prima dell'ultima data
        if i == len(true_quarters_date)-1:
            break
        
        
        #prelevo le serie ogni trimestre
        start=true_quarters_date[i]
        end=true_quarters_date[i+1]
        
        w_opt_mrkwz = compute_minimization(start, prices_df_from100)   

        rend_cum_mrkwz = investing_next_trim(start, end, prices_df_from100,w_opt_mrkwz)
        
    
        #considero 100 solo nel primo caso. Per gli altri trimestri tolgo il primo
        # valore
        if i==0:
            ce_mrkwz_tot=ce_mrkwz_tot.append(pd.Series(rend_cum_mrkwz))
    
        else:
            ce_mrkwz_tot=ce_mrkwz_tot.append(pd.Series(rend_cum_mrkwz)[1:])
        
    #creo la ce del hrp
    ce_mrkwz = np.cumprod(ce_mrkwz_tot)[:-1]
    #ce_hrp=pd.Series(ce_hrp).iloc[:-1]
    ce_mrkwz.index=index
    return ce_mrkwz


def plot_CLA_vs_HRP(ce_mrkwz, ce_hrp):
    #plotting HRP vs CLA.
    plt.figure(figsize=(cm_to_inch(40), cm_to_inch(25)))
    ce_mrkwz.plot(label = 'CLA')
    ce_hrp.plot(label = 'HRP')
    plt.title("HRP vs CLA", fontsize = 20)
    plt.xlabel(xlabel = 'Date', fontsize = 20)
    plt.ylabel(ylabel = 'Portfolio value', fontsize = 20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()
    
    #sharpe mrkwz
    print('The sharpe ratio for mrkwz is:', sharpe_ratio(ce_mrkwz.pct_change().dropna()))
    
    #sharpe hrp
    print('The sharpe ratio for hrp is:',sharpe_ratio(ce_hrp.pct_change().dropna()))












# =============================================================================
# #BUILD EQUALLY WEIGHTED CURVE
# =============================================================================

def get_eq_we_ce_from2015(prices_df_from100):
    w_eq_we = [1/prices_df_from100.shape[1]] * prices_df_from100.shape[1]
    w_eq_we = pd.DataFrame(w_eq_we).T
    w_eq_we.columns = prices_df_from100.columns
    end = pd.to_datetime('2021-07-01')
    prices_for_eq_we = prices_df_from100.loc[pd.to_datetime('2015-01-01'):end, :]
    ret_eq_we = prices_for_eq_we.pct_change().dropna()
    

    w_df_eq_we = pd.DataFrame(columns= prices_for_eq_we.columns)
    z = 0
    while z < len(ret_eq_we):
        w_df_eq_we = w_df_eq_we.append(w_eq_we)
        z+=1
    
    ret_eq_we = (w_df_eq_we.values * ret_eq_we).sum(1)
    rend_cum_eq_we = np.hstack([100,ret_eq_we + 1])
    
    ce_eq_we = np.cumprod(rend_cum_eq_we)[:-1]
    ce_eq_we = pd.Series(ce_eq_we)
    ce_eq_we.index=prices_for_eq_we.index[:-1]
    return ce_eq_we


def get_eq_we_ce_from2010(prices_df_from100):
    w_eq_we = [1/prices_df_from100.shape[1]] * prices_df_from100.shape[1]
    w_eq_we = pd.DataFrame(w_eq_we).T
    w_eq_we.columns = prices_df_from100.columns
    end = pd.to_datetime('2021-07-01')
    prices_df_from100 = prices_df_from100.loc[:end]
    ret_eq_we = prices_df_from100.pct_change().dropna()
    

    w_df_eq_we = pd.DataFrame(columns= prices_df_from100.columns)
    z = 0
    while z < len(ret_eq_we):
        w_df_eq_we = w_df_eq_we.append(w_eq_we)
        z+=1
    
    ret_eq_we = (w_df_eq_we.values * ret_eq_we).sum(1)
    rend_cum_eq_we = np.hstack([100,ret_eq_we + 1])
    
    ce_eq_we = np.cumprod(rend_cum_eq_we)[:-1]
    ce_eq_we = pd.Series(ce_eq_we)
    ce_eq_we.index=prices_df_from100.index[:-1]
    return ce_eq_we




# =============================================================================
# #TDA PART
# =============================================================================

# =============================================================================
# #estimation of the confidence values and the excess returns after views values
# =============================================================================
#importing signal df
def import_text_signal_df(path):
    text_signals=pd.read_csv(path)
    index = text_signals.iloc[:,0]
    index = pd.Series([pd.to_datetime(i) for i in index])
    text_signals = text_signals.iloc[:,1:]
    text_signals.index = index
    #deleting danone signals
    text_signals['danone'] = 0
    return text_signals


def split_pos_neg_signals(text_signals):
    pos_signals = text_signals[text_signals >= 0].fillna(0)
    neg_signals = text_signals[text_signals <= 0].fillna(0)
    return pos_signals, neg_signals



#compute the average excess return for positive and negative news with respe
def get_excess_ret_dict(text_signals, prices_df_from100):
    print("Takes a couple of minutes, wait please")

    pos_signals, neg_signals = split_pos_neg_signals(text_signals)
    
    prices_df_from100 = prices_df_from100.loc[:pd.to_datetime('2021-07-01')]
    
    ce_eq_we = get_eq_we_ce_from2010(prices_df_from100)

    annualized_ret_bench = ann_ret(ce_eq_we, 11.5)
    excess_ret_dict = {}
    for col in pos_signals.columns:
        excess_ret_dict[col] = {}
        comp_signal_pos = pos_signals.loc[:,col]
        comp_signal_neg = neg_signals.loc[:,col]
        
        #creo un df con 14 colonne e lo stesso numero di righe di total rolling
        zero_array = np.array([0]*14*len(comp_signal_pos))
        zero_array = zero_array.reshape(len(comp_signal_pos), 14)
        zero_df = pd.DataFrame(zero_array)
        zero_df.index = comp_signal_pos.index      
        comp_list_copy = list(pos_signals.columns)
        comp_list_copy.remove(col)
        zero_df.columns = comp_list_copy
        
        comp_signal_pos = pd.concat([comp_signal_pos, zero_df],1)
        comp_signal_neg = pd.concat([comp_signal_neg, zero_df],1)
        
        w_pos = from_signal_to_w_unsort(comp_signal_pos, 0.5)
        w_neg = from_signal_to_w_unsort(comp_signal_neg, 0.5)
        
        w_pos = w_pos.loc[prices_df_from100.index, :]
        w_neg = w_neg.loc[prices_df_from100.index, :]
        
        all_ret = prices_df_from100.pct_change(1).shift(-2).dropna()
        
        #riordino le colonne dell'all_ret
        all_ret_list = []
        for colu in w_pos.columns:
            all_ret_list.append(all_ret.loc[:,colu])
        
        all_ret = pd.concat(all_ret_list, 1)
            
    
        all_rend_pos = (w_pos.iloc[:-2,:].values * all_ret.values).sum(1)
        all_rend_neg = (w_neg.iloc[:-2,:].values * all_ret.values).sum(1)
        
        cum_ret_pos = np.hstack([100, all_rend_pos + 1])
        cum_ret_neg = np.hstack([100, all_rend_neg + 1])
    
        ce_pos = pd.Series(np.cumprod(cum_ret_pos))
        ce_pos.index = prices_df_from100.index[:-1]
        ce_neg = pd.Series(np.cumprod(cum_ret_neg))
        ce_neg.index = prices_df_from100.index[:-1]
            
        annualized_ret_pos = ann_ret(ce_pos, 11.5)
        annualized_ret_neg = ann_ret(ce_neg, 11.5)
        
        excess_ret_dict[col]['pos'] = annualized_ret_pos - annualized_ret_bench
        excess_ret_dict[col]['neg'] = annualized_ret_neg - annualized_ret_bench
        
    return excess_ret_dict

def get_common_days_df(signal_df, prices_df):
    signal_df_copy = pd.DataFrame(signal_df.copy())
    prices_df_copy = pd.DataFrame(prices_df.copy())
    join_df=signal_df_copy.join(prices_df_copy.iloc[:,0], rsuffix='new', how = 'outer').fillna(method='ffill')
    new_signal_df = join_df.iloc[:,:-1]
    common_days = new_signal_df.index.intersection(prices_df_copy.index)
    new_signal_df = new_signal_df.loc[common_days,:]
    return new_signal_df
    
def extend_df_dates(df_to_extend, df_with_other_dates):
    df_to_extend_copy = pd.DataFrame(df_to_extend.copy())
    df_with_other_dates_copy = pd.DataFrame(df_with_other_dates.copy())
    join_df=df_to_extend_copy.join(df_with_other_dates_copy.iloc[:,0], rsuffix='new', how = 'outer').fillna(method='ffill')
    new_df = join_df.iloc[:,:-1]
    return new_df

    


def estimate_views_confid(text_signals, prices_df_from100):
    print("Takes a couple of minutes, wait please")
    pos_signals, neg_signals = split_pos_neg_signals(text_signals)

    view_acc_dict = {}
    for col in pos_signals.columns:
        view_acc_dict[col] = {}
        comp_signal_pos = pos_signals.loc[:,col]
        index_pos = comp_signal_pos[comp_signal_pos == 1].index
        comp_signal_neg = neg_signals.loc[:,col]
        index_neg = comp_signal_neg[comp_signal_neg == -1].index
        
        #creo un df con 14 colonne e lo stesso numero di righe di total rolling
        zero_array = np.array([0]*14*len(comp_signal_pos))
        zero_array = zero_array.reshape(len(comp_signal_pos), 14)
        zero_df = pd.DataFrame(zero_array)
        zero_df.index = comp_signal_pos.index      
        comp_list_copy = list(pos_signals.columns)
        comp_list_copy.remove(col)
        zero_df.columns = comp_list_copy
        
        comp_signal_pos = pd.concat([comp_signal_pos, zero_df],1)
        comp_signal_neg = pd.concat([comp_signal_neg, zero_df],1)
        
        w_pos = from_signal_to_w_unsort(comp_signal_pos, 0.5)
        w_neg = from_signal_to_w_unsort(comp_signal_neg, 0.5)
        
        w_pos = get_common_days_df(w_pos, prices_df_from100)
        w_neg = get_common_days_df(w_neg, prices_df_from100)

        
        w_pos = w_pos.loc[prices_df_from100.index, :]
        w_neg = w_neg.loc[prices_df_from100.index, :]
        
        all_ret = prices_df_from100.pct_change(1).shift(-2).dropna()
        
        #riordino le colonne dell'all_ret
        all_ret_list = []
        for colu in w_pos.columns:
            all_ret_list.append(all_ret.loc[:,colu])
        
        all_ret = pd.concat(all_ret_list, 1)
            
    
        all_rend_pos = (w_pos.iloc[:-2,:].values * all_ret.values).sum(1)
        all_rend_neg = (w_neg.iloc[:-2,:].values * all_ret.values).sum(1)
        
        cum_ret_pos = np.hstack([100, all_rend_pos + 1])
        cum_ret_neg = np.hstack([100, all_rend_neg + 1])
    
        ce_pos = pd.Series(np.cumprod(cum_ret_pos))
        ce_pos.index = prices_df_from100.index[:-1]
        ce_neg = pd.Series(np.cumprod(cum_ret_neg))
        ce_neg.index = prices_df_from100.index[:-1]
        
        #controllo quando gli excess ret sono maggiori/minori del bench
        
        inters_pos = index_pos.intersection(ce_pos.index)
        inters_pos_subset = []
        for i in range(1,len(inters_pos)+1):
            if i-1 == 0:
                inters_pos_subset.append(inters_pos[i-1])
            elif i%15==0:
                inters_pos_subset.append(inters_pos[i-1])
            elif i-1 == len(inters_pos)-1 :
                inters_pos_subset.append(inters_pos[i-1])
        
        ret_ce_pos = ce_pos.loc[inters_pos_subset]
        ret_ce_pos = ret_ce_pos.pct_change(1).dropna()
        
        
        inters_neg = index_neg.intersection(ce_neg.index)
        inters_neg_subset = []
        for i in range(1,len(inters_neg)+1):
            if i-1 == 0:
                inters_neg_subset.append(inters_neg[i-1])
            elif i%15==0:
                inters_neg_subset.append(inters_neg[i-1])
            elif i-1 == len(inters_neg)-1 :
                inters_neg_subset.append(inters_neg[i-1])
                
        ret_ce_neg = ce_neg.loc[inters_neg_subset]
        ret_ce_neg = ret_ce_neg.pct_change(1).dropna()
        
        neg_dates_series= pd.Series(inters_neg_subset)
        neg_dates_series.index = neg_dates_series
        pos_dates_series= pd.Series(inters_pos_subset)
        pos_dates_series.index = pos_dates_series

        ce_eq_we_pos = extend_df_dates(ce_eq_we, pos_dates_series)
        ce_eq_we_neg = extend_df_dates(ce_eq_we, neg_dates_series)
        ret_ce_eq_we_pos = ce_eq_we_pos.loc[inters_pos_subset]
        ret_ce_eq_we_neg = ce_eq_we_neg.loc[inters_neg_subset]
        ret_ce_eq_we_pos = pd.Series(ret_ce_eq_we_pos.iloc[:,0].pct_change(1).dropna())
        ret_ce_eq_we_neg = pd.Series(ret_ce_eq_we_neg.iloc[:,0].pct_change(1).dropna())
        
        excess_ret_pos = ret_ce_pos.values - ret_ce_eq_we_pos.values
        excess_ret_neg = ret_ce_eq_we_neg.values - ret_ce_neg.values  
    
        excess_ret_pos[excess_ret_pos >= 0] = 1
        excess_ret_pos[excess_ret_pos < 0] = 0
        
        excess_ret_neg[excess_ret_neg >= 0] = 1
        excess_ret_neg[excess_ret_neg < 0] = 0
            
        acc_pos = excess_ret_pos.sum()/len(excess_ret_pos)
        acc_neg = excess_ret_neg.sum()/len(excess_ret_neg)
        view_acc_dict[col]['pos'] = acc_pos
        view_acc_dict[col]['neg'] = acc_neg
        
    return view_acc_dict









# =============================================================================
# #B&L + HRP Aggregation
# =============================================================================

#new exp ret formulae
def BL_exp_ret(cov, link, uncertainty_df, pi_impl_vec, views_exp_ret):
    tau = 1
    tau_eps_inv = np.linalg.inv(tau*cov)
    pT_omeg_p =  np.matmul(np.matmul(link.T.values,uncertainty_df.values),link.values)
    tau_eps_inv_pi= np.dot(tau_eps_inv, pi_impl_vec)
    pT_omeg_q = np.dot(np.matmul(link.T,uncertainty_df),views_exp_ret).reshape(15,1)
    new_exp_ret = np.matmul(np.linalg.inv(tau_eps_inv + pT_omeg_p), (tau_eps_inv_pi + pT_omeg_q))
    new_exp_ret = pd.Series(new_exp_ret.reshape(15,))
    new_exp_ret.index = pi_impl_vec.index
    return new_exp_ret
    

def from_text_to_link(daily_signal):
    daily_signal_copy = daily_signal.copy()
    link_df = pd.DataFrame(columns = daily_signal_copy.index)
    comp_with_signal= list(daily_signal_copy[daily_signal_copy.isin([1,-1])].index)
    for comp in comp_with_signal:
        daily_signal_copy2 = daily_signal_copy.copy()
        daily_signal_copy2[daily_signal_copy2.index != comp] = 0
        link_df = link_df.append(daily_signal_copy2)
    return link_df

def from_link_to_uncert_mat(daily_signal, view_acc_dict):
    daily_signal_copy = daily_signal.copy()
    comp_with_signal= daily_signal_copy[daily_signal_copy.isin([1,-1])]
    acc_val_list = []
    for comp in comp_with_signal.index:
        if comp_with_signal.loc[comp] == 1:
            acc_val_list.append(view_acc_dict[comp]['pos'])
        elif comp_with_signal.loc[comp] == -1:
            acc_val_list.append(view_acc_dict[comp]['neg'])
    uncert_mat = pd.DataFrame(np.diag(acc_val_list))
    uncert_mat.index, uncert_mat.columns = comp_with_signal.index, comp_with_signal.index
    return uncert_mat

def from_sig_to_exp_ret(daily_signal):
    daily_signal_copy = daily_signal.copy()
    comp_with_signal= daily_signal_copy[daily_signal_copy.isin([1,-1])]
    exp_ret_vec = []
    for comp,sig in zip(comp_with_signal.index, comp_with_signal.values):
        if sig ==-1:
            exp_ret_vec.append(excess_ret_dict[comp]['neg'])
        elif sig ==1:
            exp_ret_vec.append(excess_ret_dict[comp]['pos'])
    exp_ret_series = pd.Series(exp_ret_vec)
    exp_ret_series.index = comp_with_signal.index
    return exp_ret_series *100

#new var_covar_matr formula
def posterior_var(cov, link, uncertainty_df):
    tau = 1
    tau_eps_inv = np.linalg.inv(tau*cov)
    pT_omeg_p =  np.matmul(np.matmul(link.T.values,uncertainty_df.values),link.values)
    new_var = pd.DataFrame((np.linalg.inv(tau_eps_inv + pT_omeg_p).reshape(15,15)))
    new_var.index, new_var.columns = cov.index, cov.index
    return new_var

    
def new_weights_BL(new_exp_ret, risk_adv_coef, new_var_covar_mat):
    new_weights = pd.DataFrame(np.dot(new_exp_ret, np.linalg.inv(risk_adv_coef * new_var_covar_mat)))
    new_weights.index = new_var_covar_mat.index
    new_weights = (new_weights/new_weights.sum())
    return new_weights


def sort_columns(prices_df_from100,text_signals):
    sorted_list=[]
    for col in prices_df_from100.columns:
        sorted_list.append(text_signals.loc[:,col])
    sorted_df = pd.concat(sorted_list,1)
    return sorted_df





def get_TDA_allocations(prices_df_from100, text_signals,view_acc_dict):
    
    #sorting text_signals columns according to prices_df_from100 columns
    text_signals_copy = text_signals.copy()
    sorted_text_signals = sort_columns(prices_df_from100,text_signals_copy)
    
    weights_hrp_BL_df = pd.DataFrame(columns = prices_df_from100.columns)
    for i in range(len(true_quarters_date)):
        #condizione per far terminare il loop prima dell'ultima data
        if i == len(true_quarters_date)-1:
            break
        
        #prelevo le serie ogni trimestre
        start=true_quarters_date[i]
        end=true_quarters_date[i+1]
        
        #HRP part
        corr_mat = get_corr_matr(prices_df_from100, start)
        
        dist_df = compute_distances(corr_mat)
        
        euclid_df = compute_euclid_dist(dist_df)
        
        Z = clustering(euclid_df, plot=False)
        
        var_cov_mat, sorted_cov_mat = quasi_diagonalization(prices_df_from100, Z)
        
        allocations = recursive_bisection(var_cov_mat, sorted_cov_mat)
    
    
        #B&L part
        if i == 0:
            risk_coef = risk_adv_coef(ce_eq_we.pct_change(1).dropna())
        else:
            risk_coef = risk_adv_coef(ce_hrp.pct_change(1).dropna())
        #allocazioni previste dal hrp
        allocations_copy = allocations.copy().T
        
        daily_signal = sorted_text_signals.loc[start:end,:]
        for row in range(len(daily_signal)-1):
            #var covar matr, calcolata su tutti i prezzi fino allinizio del trimestre
            var_covar_mat = (prices_df_from100.loc[:daily_signal.index[row],:].pct_change(1).dropna()).cov()
            
            # FIRST FORMULA : implied returns
            impl_ret = pd.DataFrame(np.dot((risk_coef * var_covar_mat), allocations_copy))
            impl_ret.index = var_covar_mat.columns
            
            #SECOND FORMULA: calcolo gli expected return inserendo le views
            link = from_text_to_link(daily_signal.iloc[row,:])
            uncertainty_df = from_link_to_uncert_mat(daily_signal.iloc[row,:],view_acc_dict)
            exp_ret_views = from_sig_to_exp_ret(daily_signal.iloc[row,:])
    
            new_exp_ret = BL_exp_ret(var_covar_mat, link, uncertainty_df, impl_ret, exp_ret_views)
    
            #THIRD FORMULA: new variance
            new_variance = posterior_var(var_covar_mat, link, uncertainty_df)
    
            #FOURTH FORMULA: new_var_covar_matr
            new_var_covar_mat = var_covar_mat + new_variance
            
            #with the new var_covar_mat I can obtain back the new weights according to
            #the 'first formula'
            new_weights = new_weights_BL(new_exp_ret, risk_coef, new_var_covar_mat)
            new_weights.columns = [daily_signal.index[row]]
            
            weights_hrp_BL_df = weights_hrp_BL_df.append(new_weights.T)
            
    print("The turnover for TDA allocations is"+ str((np.abs(weights_hrp_BL_df.diff()).mean()).sum()*250))
            
    return weights_hrp_BL_df



def get_TDA_equity_curve(weights_hrp_BL_df, prices_df_from100):
    prices_df_ante2015,prices_df_from2015 = splitting_prices_df(prices_df_from100)
   
    all_ret_from2015 = prices_df_from2015.pct_change().shift(-2).loc[:pd.to_datetime('2021-06-30'),:]
    weights_hrp_BL_df = weights_hrp_BL_df.loc[list(all_ret_from2015.index),:]
    
    all_ret = (all_ret_from2015.values * weights_hrp_BL_df.values).sum(1)
    
    cum_ret = np.hstack([100,all_ret+1])
    ce_TDA = pd.Series(np.cumprod(cum_ret))[:-1]
    ce_TDA.index = weights_hrp_BL_df.index
    
    return ce_TDA
    

    
    
    
    
def plot_ce_comparison(ce_TDA, ce_hrp, ce_eq_we, ce_mrkwz):
    ce_eq_we_from2015 = ce_eq_we.loc[pd.to_datetime('2015-01-01'):pd.to_datetime('2021-06-30')]
    ce_eq_we_from2015 = 100 * ce_eq_we_from2015/ce_eq_we_from2015.iloc[0]

    
    plt.figure(figsize=(cm_to_inch(40), cm_to_inch(25)))
    ce_TDA.plot(label = 'ce_TDA', color = 'b')
    ce_hrp.plot(label = 'ce_hrp', color = 'r')
    ce_mrkwz.plot(label = 'mrkwz', color = 'g')
    ce_eq_we_from2015.plot(label= 'eq_we', color = 'y')
    plt.legend(fontsize = 15)
    plt.title('Performance comparison', fontsize = 20)
    plt.xlabel(xlabel = 'Date', fontsize = 20)
    plt.ylabel(ylabel = 'Portfolio value', fontsize = 20)
    plt.grid()
    plt.show()
    
    print("The Sharpe ratio for ce_TDA is:" +str(sharpe_ratio(ce_TDA.pct_change().dropna())))
    
    
    
    
    



#MAIN CODE THAT INCLUDES HRP, MARKOWITZ AND TDA
# =============================================================================
# # PURE-HRP PART
# =============================================================================
prices_df_from100 = create_prices_df_from100(comp_list_names, comp_list_prices)  
prices_df_pre2015,prices_df_from2015=splitting_prices_df(prices_df_from100)
true_quarters_date = get_trimester_dates(prices_df_from2015)

ce_hrp_tot=pd.Series()

#prices until 2021-06-30
index = prices_df_from2015.index[:-21]

for i in range(len(true_quarters_date)):
    #condizione per far terminare il loop prima dell'ultima data
    if i == len(true_quarters_date)-1:
        break
    
    #prelevo le serie ogni trimestre
    start=true_quarters_date[i]
    end=true_quarters_date[i+1]
    
    corr_mat = get_corr_matr(prices_df_from100, start)
    
    dist_df = compute_distances(corr_mat)
    
    euclid_df = compute_euclid_dist(dist_df)
    
    Z = clustering(euclid_df, plot=False)
    
    var_cov_mat, sorted_cov_mat = quasi_diagonalization(prices_df_from100, Z)
    
    allocations = recursive_bisection(var_cov_mat, sorted_cov_mat)
    
    trimester_prices = prices_df_from100.loc[start:end,:]
    
    ce_hrp, rend_cum_hrp = get_equity_curve(trimester_prices, allocations)
    
    #considero 100 solo nel primo caso. Per gli altri trimestri tolgo il primo
    # valore
    if i==0:
        ce_hrp_tot=ce_hrp_tot.append(pd.Series(rend_cum_hrp))

    else:
        ce_hrp_tot=ce_hrp_tot.append(pd.Series(rend_cum_hrp)[1:])


#creo la ce del hrp
ce_hrp = np.cumprod(ce_hrp_tot)[:-1]
ce_hrp=pd.Series(ce_hrp)
ce_hrp.index=index   


# =============================================================================
# #MARKOWITZ PART
# =============================================================================
ce_mrkwz = compute_CLA(prices_df_from100)
plot_CLA_vs_HRP(ce_mrkwz, ce_hrp)



# =============================================================================
# #EQUALLY WEIGHTED BENCHMARK
# =============================================================================
ce_eq_we = get_eq_we_ce_from2010(prices_df_from100)



# =============================================================================
# #TDA PART
# =============================================================================

# 1. PARAMETERS ESTIMATION (THE PART THAT TAKES LONGER)
path = r'C:\Users\111949\OneDrive\Desktop\text_signals.csv' #insert own text_signals_path
text_signals = import_text_signal_df(path)
excess_ret_dict = get_excess_ret_dict(text_signals, prices_df_from100)
view_acc_dict = estimate_views_confid(text_signals, prices_df_from100)


# 2. HRP+B&L AGGREGATION
weights_TDA = get_TDA_allocations(prices_df_from100, text_signals,view_acc_dict)
ce_TDA = get_TDA_equity_curve(weights_TDA, prices_df_from100)

# 3. PLOT COMPARISON
plot_ce_comparison(ce_TDA, ce_hrp, ce_eq_we, ce_mrkwz)







