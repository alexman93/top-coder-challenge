import numpy as np
import pandas as pd
import pickle
import pandas as pd
import json
from sklearn.preprocessing import add_dummy_feature 
from sklearn.linear_model import LinearRegression, RANSACRegressor


def matpred(X,Y,nt,typemat,nodefeatsmat,centermat,stdmat,coefmat,splitvalmat,mapmat,featmat,ymat):
    for t in range(X.shape[0]):
        # print(t)
        for i in range(nt):
            curnode=0
            while typemat[i,curnode]!=2:
                cnstart=curnode
                if typemat[i,cnstart]==0:
                    Xs=(X[t,nodefeatsmat[i,curnode,:]]-centermat[i,curnode,:])/stdmat[i,curnode,:]
                    feat=np.sum(Xs*coefmat[i,curnode,:-1])+coefmat[i,curnode,-1]
                    if feat>=splitvalmat[i,curnode]:
                        curnode=mapmat[i,curnode,1]
                    else:
                        curnode=mapmat[i,curnode,0]
                if typemat[i,cnstart]==1:
                    feat=X[t,nodefeatsmat[i,curnode,:]][featmat[i,curnode]]
                    # feat=Xs[featmat[i,curnode]]
                    if feat>=splitvalmat[i,curnode]:
                        curnode=mapmat[i,curnode,1]
                    else:
                        curnode=mapmat[i,curnode,0]
            Y[t]+=ymat[i,curnode]/nt

with open("flformat.pkl", "rb") as f:
    Gfl=pickle.load(f)
    
with open('public_cases.json', 'r') as file:
       raw_data = json.load(file)
       # print(data)

       
records = []
for item in raw_data:
    row = item["input"].copy()           # grab the three inputs
    row["expected_output"] = item["expected_output"]
    records.append(row)

# ---- 3. Build the DataFrame ------------------------------------------------
df = pd.DataFrame(records,
                  columns=["trip_duration_days",
                           "miles_traveled",
                           "total_receipts_amount",
                           "expected_output"])

df["milesperday"]=df["miles_traveled"]/df["trip_duration_days"]
df["receiptsperday"]=df["total_receipts_amount"]/df["trip_duration_days"]

with open('private_cases.json', 'r') as file:
       raw_data = json.load(file)
       # print(data)

# hoog      
records = []
for item in raw_data:
    # row = item["input"].copy()           # grab the three inputs
    # row["expected_output"] = item["expected_output"]
    records.append(item)

# ---- 3. Build the DataFrame ------------------------------------------------
dfpri = pd.DataFrame(records,
                  columns=["trip_duration_days",
                           "miles_traveled",
                           "total_receipts_amount"])

df["milesperday"]=df["miles_traveled"]/df["trip_duration_days"]
df["receiptsperday"]=df["total_receipts_amount"]/df["trip_duration_days"]
dfpri["milesperday"]=dfpri["miles_traveled"]/dfpri["trip_duration_days"]
dfpri["receiptsperday"]=dfpri["total_receipts_amount"]/dfpri["trip_duration_days"]

X=np.array(df.iloc[:,[0,1,2,4,5]])
Xpri=np.array(dfpri.iloc[:,[0,1,2,3,4]])
Y=np.array(X[:,3])
# hoog
pfeatures=[]
for num in range(3):
    # num=2
    pf=np.flip(np.polyfit(X[:,num],Y,4))
    pfeat=np.zeros(len(Y))
    pfeatpri=np.zeros(len(dfpri))
    for t in range(len(pf)):
        pfeat+=pf[t]*X[:,num]**t
        pfeatpri+=pf[t]*Xpri[:,num]**t
    # plt.scatter(X[:,num],Y)
    # plt.scatter(X[:,num],pfeat)
    df[str(num)]=pfeat
    dfpri[str(num)]=pfeatpri
    pfeatures.append(pf)
# hoog

pickle.dump(pfeatures, open( "pf.pkl", "wb" ) )
# hoog
# X=np.array(df.iloc[:,[0,1,2,4,5,6,7,8]])
# X=np.array(df.iloc[:,-3:])
# X1=np.hstack([X, np.ones((X.shape[0], 1))])
Y=df.iloc[:,3]
Y=np.array(Y)

# for num in range(3):
#     # num=2
#     pf=np.flip(np.polyfit(X[:,num],Y,4))
#     pfeat=np.zeros(len(Y))
#     for t in range(len(pf)):
#         pfeat+=pf[t]*X[:,num]**t
#     # plt.scatter(X[:,num],Y)
#     # plt.scatter(X[:,num],pfeat)
#     df[str(num)]=pfeat
# hoog
    
# X=np.array(df.iloc[:,[0,1,2]])
X=np.array(df.iloc[:,[0,1,2,4,5,6,7,8]])
Xpri=np.array(df.iloc[:,[0,1,2,3,4,5,6,7]])

Xalt=X.copy()
Xprialt=Xpri.copy()
Xn=X
Yn=Y
ct=1
ctind=np.zeros(len(Yn))
liscoef=[]
for t in range(12):
    Xn=X[ctind==0,:]
    Yn=Y[ctind==0]
    base_est = LinearRegression()  
    ransac   = RANSACRegressor(estimator=base_est,
                                min_samples=10,
                                residual_threshold=50, # ε
                                loss='absolute_error',    # MAE-like
                                max_trials=400,
                                random_state=0)
    ransac.fit(Xn, Yn)
    # liscoef.append(ransac.estimator_.coef_)
    # hoog
    ll1=ctind==0
    # ll2=ransac.inlier_mask_*ct
    # lll=ll1 & ll2
    ctind[ll1]=ransac.inlier_mask_*ct
    # hoog
    pred=ransac.predict(Xn[ransac.inlier_mask_,:])
    act=Yn[ransac.inlier_mask_]
    # plt.scatter(pred,act)
    
    ct+=1
    l1=~ransac.inlier_mask_
    # Xn=Xn[l1,:]
    # Yn=Yn[l1]
    print(Xn.shape)
    cf=ransac.estimator_.coef_
    inter=ransac.estimator_.intercept_
    ad=np.dot(X,cf)+inter
    ad=ad.reshape(-1,1)
    Xalt=np.hstack([Xalt,ad])
    
    adpri=np.dot(Xpri,cf)+inter
    adpri=adpri.reshape(-1,1)
    Xprialt=np.hstack([Xprialt,adpri])
    # hoog
    liscoef.append([cf,inter])
    # hoog

pickle.dump(liscoef, open( "ransac.pkl", "wb" ) )
X=Xalt
Xpri=Xprialt
Gsim=ForestReg(n_trees=100,mtry=X.shape[1],mtry2=X.shape[1],bootFrac=None,bootPart=True,partSS='Block',
          minObs=30,maxObsLin=100000,maxObsSplit=1000) 
Gsim.fit(X,Y,X,np.arange(len(Y)))
Gsim.bootPred(X,Y,X)
predpri=Gsim.predict(Xpri,Xpri)
hoog
yind=np.zeros(1)
xind=np.zeros(3)

matpred(X[0,:].reshape(1,-1),yind,100,Gfl[0],Gfl[1],Gfl[2],Gfl[3],Gfl[4],Gfl[5],Gfl[6],Gfl[7],Gfl[8])