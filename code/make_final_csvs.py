# What's the minimum RMSD of various subsamples?
import pandas as pd
import json
import os
import numpy as np

np.random.seed(seed=1)

def median(l):
    l.sort()
    if len(l) % 2 == 0:
        return (l[len(l)//2] + l[len(l)//2-1])/2
    else:
        return l[len(l)//2]

# df = pd.read_pickle('all_data_scores.pkl')
df = pd.read_csv('new_all_data_scores.csv')

scores = ['pred', 'score', 'rnascore', 'rasp', 'rosetta_hires_2010', 'rosetta_lores_2007', 'simrna']
labels = [2021, 2020, 2015, 2011, 2010, 2007, 2016]

# Calculate "overall" table.
bo1val = []
for scorename in scores:
    bo1s = median([min(
            df[df.rna_name == name].nsmallest(1, scorename)['rms'].values.tolist()
        ) for name in df.rna_name.unique()])
    bo1val.append(bo1s)

bo10val = []
for scorename in scores:
    bo10s = median([min(
            df[df.rna_name == name].nsmallest(10, scorename)['rms'].values.tolist()
        ) for name in df.rna_name.unique()])
    bo10val.append(bo10s)

print('bo1_med,bo10_med,year')
for bo1,bo10,year in zip(bo1val, bo10val, labels):
    print(f'{bo1:.4f},{bo10:.4f},{year}')


# Process each resampled dataframe into bo1/bo10 statistics and cache
# for speed in subsequent runs.
objs = []
if os.path.exists('bootstrap_cache.json'):
    with open('bootstrap_cache.json') as f: objs = json.loads(f.read())
else:
    for name in df.rna_name.unique(): 
        obj = {'name': name}
        mdf = df[df.rna_name == name].copy()
        dfs = [mdf.sample(n=5000, replace=True, random_state=ii, axis=0) for ii in range(200)]
        #replace=True表示有放回抽样，即bootstrap
        for scorename in scores:

            bo1s = [min(d.nsmallest(1, scorename)['rms'].values.tolist()) for d in dfs]
            bo10s = [min(d.nsmallest(10, scorename)['rms'].values.tolist()) for d in dfs]
            #这里bo1s长度为200
            obj['bo1_{}'.format(scorename)] = bo1s 
            obj['bo10_{}'.format(scorename)] = bo10s

        objs.append(obj)

        with open('bootstrap_cache.json', 'w') as f: f.write(json.dumps(objs))


# Make 20,000 random choices of bo1/bo10 per case and take the median.
# Accumulate statistics: median and matplotlib-formatted 95% CI.
# CI = Confidence Interval 置信区间，Pr(c1<=μ<=c2)=1-α，α是显著性水平significance level，显著性水平是估计总体参数落在某一区间内，可能犯错误的概率
nsamples = 20000
bo1vals, bo1val, bo1err = [], [], [[],[]]
for scorename in scores:

    vals = sorted([median([np.random.choice(obj['bo1_{}'.format(scorename)]) for obj in objs]) for ii in range(nsamples)])
    bo1vals.append(vals)
    bo1val.append(vals[nsamples//2-1]/2.0+vals[nsamples//2]/2.0)
    bo1err[0].append(bo1val[-1]/2.0-vals[nsamples//40]/2.0) #//40是取第2.5%的位置
    bo1err[1].append(vals[nsamples*39//40]/2.0-bo1val[-1]/2.0) #后2.5%

bo10val, bo10err = [], [[],[]]
for scorename in scores:

    vals = sorted([median([np.random.choice(obj['bo10_{}'.format(scorename)]) for obj in objs]) for ii in range(nsamples)])
    bo10val.append(vals[nsamples//2-1]/2.0+vals[nsamples//2]/2.0)
    bo10err[0].append(bo10val[-1]/2.0-vals[nsamples//40]/2.0)
    bo10err[1].append(vals[nsamples*39//40]/2.0-bo10val[-1]/2.0)


print('\n')
print('bo10_high_err,bo10_low_err,bo10_med,bo1_high_err,bo1_low_err,bo1_med,year')
for bo10_high_err, bo10_low_err, bo10_med, bo1_high_err, bo1_low_err, bo1_med, year in zip(
    bo10err[1], bo10err[0], bo10val, bo1err[1], bo1err[0], bo1val, labels
):
    print(f'{bo10_high_err:.3f},{bo10_low_err:.3f},{bo10_med:.3f},{bo1_high_err:.3f},{bo1_low_err:.3f},{bo1_med:.3f},{year}')