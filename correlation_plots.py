import os
import uproot
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn import metrics
from matplotlib import pyplot
import pickle
import seaborn as sns

def get_df(path, branches, sig):
	iters = []
	for data in uproot.pandas.iterate(path, "Friends", branches=branches, flatten=False):

		data = data[data.nleadingLepton == 1]#one leading lepton in event
		data = data[data.nsubleadingLepton == 1]#one subleading lepton in event
		data = data[data.nselectedJets_nominal > 0]#at least one jet in event
		data = data[data.dilepton_mass < 80.]#apply CR cut
		data = data[data.EventObservables_nominal_met < 100.]#apply CR cut
		#data = data[data.lepJet_nominal_deltaR < 0.4]#DeltaR cut
		data = data[data.dilepton_charge == -1]#Dirac samples only, i.e. opposite sign leptons
		#data = data[data.IsoMuTrigger_flag == True]#muon trigger applied for event

		iters.append(data)
		# add cuts

	return pd.concat(iters)

path1 = "/vols/cms/jd918/LLP/CMSSW_10_2_18/"
path = "/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/30Jul20/"

array_preselection = [
			"nleadingLepton", "nlepJet_nominal", "nsubleadingLepton", "lepJet_nominal_deltaR",
			"dilepton_charge", "IsoMuTrigger_flag"
			]

with open(os.path.join(path1,"src/PhysicsTools/NanoAODTools/data/bdt/bdt_inputs.txt")) as f:
	array_bdt = [line.rstrip() for line in f]

array_list = array_preselection + array_bdt

n_events=200000

sig_df = get_df(os.path.join(path,"/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/30Jul20/2016//HNL_*/nano_*_Friend.root"), array_list, True)
print(sig_df.shape[0])

useOneFile = False

if useOneFile:
	sig_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/HNL_dirac_all_ctau1p0e01_massHNL4p5_Vall4p549e-03-2016/nano_1_Friend.root"), array_list + ["LHEWeights_coupling_12"], True)
	wjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/WToLNu_0J_13TeV-amcatnloFXFX-pythia8-2016/nano_1_Friend.root"), array_list, False)
	tt_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/TTToSemiLeptonic_*/nano_1_Friend.root"), array_list, False)
	dyjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-2016/nano_1_Friend.root"), array_list, False)
	#bkg_df = pd.concat([wjets_df, dyjets_df, tt_df]).sample(n=n_events)
else:
	sig_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016//HNL_*/nano_*_Friend.root"), array_list, True)
	wjets_df = get_df(os.path.join(path,"2016/WToLNu_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	tt_df = get_df(os.path.join(path,"2016/TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	dyjets_df = get_df(os.path.join(path,"2016/DYJetsToLL*amcatnlo*/nano_*1_Friend.root"), array_list, False).sample(n=n_events)
'''
sig_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016//HNL_*/nano_*_Friend.root"), array_list, True)
print(sig_df.shape[0])

wjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/WToLNu_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
tt_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
dyjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/DYJetsToLL*amcatnlo*/nano_*1_Friend.root"), array_list, False).sample(n=n_events)


sig_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/HNL_dirac_all_ctau1p0e01_massHNL4p5_Vall4p549e-03-2016/nano_1_Friend.root"), array_list + ["LHEWeights_coupling_12"], True)
wjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/WToLNu_0J_13TeV-amcatnloFXFX-pythia8-2016/nano_1_Friend.root"), array_list, False)
tt_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/TTToSemiLeptonic_*/nano_1_Friend.root"), array_list, False)
dyjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-2016/nano_1_Friend.root"), array_list, False)
#bkg_df = pd.concat([wjets_df, dyjets_df, tt_df]).sample(n=n_events)
'''

bkg_df = pd.concat([wjets_df, dyjets_df, tt_df])
print(bkg_df.shape[0])

sig_label = np.ones(sig_df.shape[0])
bkg_label = np.zeros(bkg_df.shape[0])

print("sig_df")
print(list(sig_df))

print("bkg_df")
print(list(bkg_df))

df = pd.concat([sig_df[array_bdt], bkg_df[array_bdt]])
label = np.concatenate([sig_label, bkg_label])

print(df)

for feat in array_bdt:
	if df[feat].dtypes == object:
		#print(df[feat])
		df[feat] = df[feat].map(lambda x: x[0])
		#print(df[feat])

print(df)

fig, ax = pyplot.subplots(figsize=(12,10))
pyplot.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.2)
pyplot.title('Confusion matrix')
cor = df.corr()
g = sns.heatmap(cor, annot=True, cmap=pyplot.cm.Reds, fmt=".2f")
pyplot.savefig('BDT_confusion_matrix.pdf')
pyplot.savefig('BDT_confusion_matrix.png')
#pyplot.show()

fig, ax = pyplot.subplots(figsize=(12,10))
pyplot.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.2)
pyplot.title('Correlation matrix: signal')
cor = df[label==1].corr()
g = sns.heatmap(cor, annot=True, cmap=pyplot.cm.Reds, fmt=".2f")
pyplot.savefig('BDT_confusion_matrix_sig.pdf')
pyplot.savefig('BDT_confusion_matrix_sig.png')
#pyplot.show()

fig, ax = pyplot.subplots(figsize=(12,10))
pyplot.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.2)
pyplot.title('Correlation matrix: background')
cor = df[label==0].corr()
g = sns.heatmap(cor, annot=True, cmap=pyplot.cm.Reds, fmt=".2f")
pyplot.savefig('BDT_confusion_matrix_bkg.pdf')
pyplot.savefig('BDT_confusion_matrix_bkg.png')
#pyplot.show()
'''
#iris = sns.load_dataset(\"iris\")
g = sns.pairplot(df, vars = df.keys())
g.savefig('BDT_feature_correlations.pdf')
g.savefig('BDT_feature_correlations.png')

g = sns.pairplot(df[label==1], vars = df.keys())
g.savefig('BDT_feature_correlations_sig.pdf')
g.savefig('BDT_feature_correlations_sig.png')

g = sns.pairplot(df[label==0], vars = df.keys())
g.savefig('BDT_feature_correlations_bkg.pdf')
g.savefig('BDT_feature_correlations_bkg.png')
'''
