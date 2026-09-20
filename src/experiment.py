"""Compare word, character and combined TF-IDF on BANKING77."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]

def make_model(kind, seed):
    common = dict(lowercase=True, sublinear_tf=True, min_df=1, norm='l2')
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, **common)
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=80000, **common)
    if kind == 'word': features = word
    elif kind == 'char': features = char
    else:
        features = Pipeline([('union', FeatureUnion([('word', word), ('char', char)])),
                             ('normalize', Normalizer())])
    return Pipeline([('features', features), ('clf', LinearSVC(C=1.0, dual=True,
                    tol=1e-4, max_iter=5000, random_state=seed))])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run one sampling seed and save to results_quick.')
    args = parser.parse_args()
    out = ROOT / ('results_quick' if args.quick else 'results')
    out.mkdir(exist_ok=True)
    data = ROOT / 'data'; data.mkdir(exist_ok=True)
    for split in ['train', 'test']:
        path = data / f'{split}.csv'
        if not path.exists():
            urllib.request.urlretrieve('https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/' + path.name, path)
    raw = pd.read_csv(data/'train.csv'); test = pd.read_csv(data/'test.csv')
    # Deduplication is based only on official training data. Keep test unchanged.
    raw['row_id'] = np.arange(len(raw))
    raw['key'] = raw.text.str.lower().str.split().str.join(' ')
    conflicting = raw.groupby('key').category.nunique()
    bad = set(conflicting[conflicting > 1].index)
    clean = raw[~raw.key.isin(bad)].drop_duplicates('key').copy()
    train, valid = train_test_split(clean, test_size=.2, stratify=clean.category, random_state=2026)
    labels = sorted(raw.category.unique())
    assert set(train.category) == set(valid.category) == set(test.category)
    assert not (set(train.key) & set(valid.key))
    stats = {'official_train':len(raw), 'official_test':len(test), 'classes':len(labels),
             'clean_train':len(clean), 'training_pool':len(train), 'validation':len(valid),
             'removed_train_rows':len(raw)-len(clean), 'conflicting_texts':len(bad),
             'train_pool_min_per_class':int(train.category.value_counts().min()),
             'train_pool_max_per_class':int(train.category.value_counts().max()),
             'official_test_min_per_class':int(test.category.value_counts().min()),
             'official_test_max_per_class':int(test.category.value_counts().max()),
             'mean_train_words':float(raw.text.str.split().str.len().mean()),
             'mean_test_words':float(test.text.str.split().str.len().mean()),
             'median_train_words':float(raw.text.str.split().str.len().median()),
             'cross_split_normalized_overlap':len(set(clean.key)&set(test.text.str.lower().str.split().str.join(' '))),
             'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [data/'train.csv',data/'test.csv']}}
    (out/'dataset_stats.json').write_text(json.dumps(stats,indent=2))
    train[['row_id']].to_csv(out/'train_pool_ids.csv',index=False)
    valid[['row_id']].to_csv(out/'validation_ids.csv',index=False)
    rows=[]; seed_list=[42] if args.quick else [42,43,44]
    for budget in [10,30,60,'pool']:
        for seed in ([42] if budget=='pool' else seed_list):
            # Independent of model type; first k positions yield nested subsets.
            sub = train if budget=='pool' else pd.concat([g.sample(frac=1,random_state=seed).head(budget) for _,g in train.groupby('category',sort=True)])
            sub[['row_id']].to_csv(out/f'ids_{budget}_{seed}.csv',index=False)
            for kind in ['word','char','combined']:
                model=make_model(kind,seed)
                t=time.perf_counter();model.fit(sub.text,sub.category);fit_s=time.perf_counter()-t
                vp=model.predict(valid.text)
                t=time.perf_counter();pred=model.predict(test.text);pred_s=time.perf_counter()-t
                row=dict(budget=str(budget),seed=seed,model=kind,n_train=len(sub),
                         val_accuracy=accuracy_score(valid.category,vp),
                         accuracy=accuracy_score(test.category,pred),
                         macro_f1=f1_score(test.category,pred,average='macro'),fit_seconds=fit_s,predict_seconds=pred_s)
                rows.append(row);pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
                print(row,flush=True)
                pd.DataFrame({'text':test.text,'true':test.category,'pred':pred}).to_csv(out/f'predictions_{budget}_{seed}_{kind}.csv',index=False)
                if budget=='pool':
                    joblib.dump(model,out/f'model_{kind}.joblib',compress=3)
                    pd.DataFrame(classification_report(test.category,pred,output_dict=True,zero_division=0)).T.to_csv(out/f'per_class_{kind}.csv')
    metrics=pd.DataFrame(rows)
    summary=metrics.groupby(['budget','model'],sort=False).agg(accuracy=('accuracy','mean'),accuracy_std=('accuracy','std'),macro_f1=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),fit_seconds=('fit_seconds','mean'),runs=('seed','count')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.4,4))
    for kind,label in [('word','Word TF-IDF'),('char','Character TF-IDF'),('combined','Combined TF-IDF')]:
        s=summary[(summary.model==kind)&(summary.budget!='pool')].copy();s['k']=s.budget.astype(int);s=s.sort_values('k')
        ax.errorbar(s.k,s.accuracy*100,yerr=s.accuracy_std.fillna(0)*100,marker='o',capsize=4,label=label)
    ax.set(xlabel='Maximum training examples per class',ylabel='Test accuracy (%)',xticks=[10,30,60]);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=180);plt.close(fig)
    majority=train.category.value_counts().idxmax()
    base={'majority_label':majority,'accuracy':float((test.category==majority).mean()),'macro_f1':float(f1_score(test.category,[majority]*len(test),average='macro'))}
    (out/'dummy.json').write_text(json.dumps(base,indent=2))
    a=pd.read_csv(out/'predictions_pool_42_word.csv'); b=pd.read_csv(out/'predictions_pool_42_combined.csv')
    d=(b.pred==b.true).astype(int).to_numpy()-(a.pred==a.true).astype(int).to_numpy()
    rng=np.random.default_rng(2026)
    boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(2000)])
    inference={'comparison':'combined minus word, fixed pool models','delta_accuracy':float(d.mean()),'bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'resamples':2000,'note':'Paired test-row bootstrap; conditional on fitted models, not training variability.'}
    (out/'comparison.json').write_text(json.dumps(inference,indent=2))
    errors=b[b.pred!=b.true].copy();errors.to_csv(out/'errors_combined.csv',index=False)
    errors.groupby(['true','pred']).size().sort_values(ascending=False).head(15).rename('count').to_csv(out/'top_confusions.csv')
    test_keys=test.text.str.lower().str.split().str.join(' ')
    nonoverlap=~test_keys.isin(set(clean.key))
    sensitivity={k:float((pd.read_csv(out/f'predictions_pool_42_{k}.csv').pred[nonoverlap].to_numpy()==test.category[nonoverlap].to_numpy()).mean()) for k in ['word','char','combined']}
    sensitivity['n_test_nonoverlap']=int(nonoverlap.sum())
    (out/'overlap_sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
    env={'python':sys.version,'platform':platform.platform(),'cpu':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor(),'logical_cpus':os.cpu_count(),
         'packages':{m.__name__:m.__version__ for m in [sklearn,np,pd,scipy,joblib,matplotlib]},'seed_list':seed_list,'split_seed':2026,'C':1.0}
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
