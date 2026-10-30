import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, json
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix, roc_auc_score,
    average_precision_score, ConfusionMatrixDisplay, precision_recall_curve)
df=pd.read_csv('/mnt/user-data/uploads/dataset.csv')
META=['segment','anomaly','train','channel']
feats=[c for c in df.columns if c not in META]
X=df[feats].copy()
skewed=['mean','var','std','diff_var','diff2_var','var_div_duration','var_div_len']
for c in ['var','diff_var','diff2_var','var_div_duration','var_div_len','std']: X[c]=np.log10(X[c])
X['mean']=np.sign(X['mean'])*np.log1p(np.abs(X['mean'])*1e6)
for c in ['len','duration','gaps_squared','len_weighted','n_peaks','diff_peaks','diff2_peaks']: X[c]=np.log1p(X[c])
tr=df.train==1; te=~tr
sc=RobustScaler().fit(X[tr]); Z=sc.transform(X)
y=df.anomaly.values
res={}
def ev(name,score,thr):
    pred=(score>=thr).astype(int)
    p,r,f,_=precision_recall_fscore_support(y[te],pred[te],average='binary',zero_division=0)
    cm=confusion_matrix(y[te],pred[te])
    res[name]=dict(roc=roc_auc_score(y[te],score[te]),ap=average_precision_score(y[te],score[te]),p=p,r=r,f1=f,cm=cm.tolist(),flag=int(pred.sum()))
    return pred
# Isolation Forest
iso=IsolationForest(n_estimators=300,contamination=0.2,random_state=42).fit(Z[tr])
s_if=-iso.score_samples(Z); thr_if=np.quantile(s_if[tr],0.8)
p_if=ev('IF',s_if,thr_if)
lof=LocalOutlierFactor(n_neighbors=35,novelty=True,contamination=0.2).fit(Z[tr])
s_lof=-lof.score_samples(Z); thr_lof=np.quantile(s_lof[tr],0.8)
p_lof=ev('LOF',s_lof,thr_lof)
# rolling-style baseline: max robust |z| per feature (channel-wise)
Zc=pd.DataFrame(Z,columns=feats); g=df.channel.values
zs=np.zeros(len(df))
for ch in np.unique(g):
    m=(g==ch); mt=m&tr.values
    med=np.median(Z[mt],0); mad=np.median(np.abs(Z[mt]-med),0)*1.4826+1e-9
    zs[m]=np.sqrt((((Z[m]-med)/mad)**2).clip(0,100).mean(1))
thr_z=np.quantile(zs[tr],0.8); p_z=ev('ChannelZ',zs,thr_z)
# per-channel IF variant
s_ch=np.zeros(len(df))
for ch in np.unique(g):
    m=(g==ch); mt=m&tr.values
    if mt.sum()<30: s_ch[m]=s_if[m]; continue
    mdl=IsolationForest(n_estimators=300,random_state=42).fit(Z[mt]); s_ch[m]=-mdl.score_samples(Z[m])
thr_ch=np.quantile(s_ch[tr],0.8); ev('IF_perchannel',s_ch,thr_ch)
for k,v in res.items(): print(k,{a:(round(b,3) if not isinstance(b,list) else b) for a,b in v.items()})
df['score']=s_if; df['pred']=p_if
print('flagged overall',p_if.sum(),p_if.mean())
print(df.groupby('channel').agg(n=('pred','size'),flag=('pred','sum'),true=('anomaly','sum'),score=('score','mean')).round(3))
print(df.groupby('anomaly').score.describe().round(3))
# correlation
corr=df[feats].corr(); print(corr.abs().where(~np.eye(len(corr),dtype=bool)).stack().sort_values(ascending=False).head(8))
print(df.groupby('sampling').anomaly.mean())
# figures
sns.set_theme(style='whitegrid')
fig,ax=plt.subplots(1,2,figsize=(10,4))
df.channel.value_counts().plot.bar(ax=ax[0],color='#3b6ea5');ax[0].set_title('Segments per channel');ax[0].set_ylabel('Count')
pd.crosstab(df.channel,df.anomaly).plot.bar(stacked=True,ax=ax[1],color=['#7fb3d5','#c0392b']);ax[1].set_title('Labelled class by channel');ax[1].legend(['Nominal','Anomaly'])
plt.tight_layout();plt.savefig('figs/fig1_channels.png',dpi=150);plt.close()
fig,ax=plt.subplots(2,3,figsize=(12,6))
for a,c in zip(ax.ravel(),['std','kurtosis','skew','duration','n_peaks','diff_peaks']):
    sns.histplot(data=df,x=c,hue=df.anomaly.map({0:'Nominal',1:'Anomaly'}),bins=40,ax=a,log_scale=(c in['std','duration']),element='step');a.set_title(c)
plt.tight_layout();plt.savefig('figs/fig2_hist.png',dpi=150);plt.close()
fig,ax=plt.subplots(1,3,figsize=(12,4))
for a,c in zip(ax,['std','kurtosis','duration']):
    sns.boxplot(data=df,x='anomaly',y=c,ax=a,log_scale=(c in['std','duration']),palette=['#7fb3d5','#c0392b'],hue='anomaly',legend=False);a.set_title(c)
plt.tight_layout();plt.savefig('figs/fig3_box.png',dpi=150);plt.close()
plt.figure(figsize=(9,7.5));sns.heatmap(corr,cmap='coolwarm',center=0,annot=True,fmt='.1f',annot_kws={'size':6});plt.title('Pearson correlation of segment features');plt.tight_layout();plt.savefig('figs/fig4_corr.png',dpi=150);plt.close()
plt.figure(figsize=(7,4));sns.histplot(x=s_if[te],hue=df.anomaly[te].map({0:'Nominal',1:'Anomaly'}),bins=40,element='step');plt.axvline(thr_if,color='k',ls='--',label='threshold');plt.xlabel('Isolation Forest anomaly score');plt.title('Score distribution (test split)');plt.tight_layout();plt.savefig('figs/fig5_scores.png',dpi=150);plt.close()
fig,ax=plt.subplots(1,3,figsize=(13,4))
for a,(k,pp) in zip(ax,[('IF',p_if),('LOF',p_lof),('ChannelZ',p_z)]):
    ConfusionMatrixDisplay(confusion_matrix(y[te],pp[te]),display_labels=['Nominal','Anomaly']).plot(ax=a,colorbar=False,cmap='Blues');a.set_title(k+' (test)')
plt.tight_layout();plt.savefig('figs/fig6_cm.png',dpi=150);plt.close()
plt.figure(figsize=(6,4.5))
for k,s in [('Isolation Forest',s_if),('LOF',s_lof),('Channel robust-z',zs)]:
    p,r,_=precision_recall_curve(y[te],s[te]);plt.plot(r,p,label=k)
plt.axhline(y[te].mean(),color='grey',ls=':',label='prevalence');plt.xlabel('Recall');plt.ylabel('Precision');plt.legend();plt.title('Precision-recall (test)');plt.tight_layout();plt.savefig('figs/fig7_pr.png',dpi=150);plt.close()
plt.figure(figsize=(9,4));d=df.copy();d['flag']=p_if
t=d.groupby('channel').agg(flag=('flag','mean'),true=('anomaly','mean'));t.plot.bar(color=['#e67e22','#c0392b']);plt.ylabel('Proportion');plt.title('Flagged vs labelled anomaly proportion by channel');plt.tight_layout();plt.savefig('figs/fig8_chan_flags.png',dpi=150);plt.close()
json.dump(res,open('res.json','w'))
