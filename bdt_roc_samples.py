import os
import uproot
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, Booster
from xgboost import plot_importance
import sklearn
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
		print(data.keys())

		data = data[data.nleadingLeptons == 1]#one leading lepton in event
		data = data[data.nsubleadingLeptons == 1]#one subleading lepton in event
		data = data[data.nselectedJets_nominal > 0]#at least one jet in event
		data = data[data.dilepton_mass < 80.]#apply CR cut
		data = data[data.dilepton_mass > 20.0]
		data = data[data.EventObservables_nominal_met < 100.]#apply CR cut
		data = data[data.MET_filter == 1]#apply MET filter

		data.lepJet_nominal_deltaR = data.lepJet_nominal_deltaR.map(lambda x: x[0])
		data = data[data.lepJet_nominal_deltaR < 2.0]#DeltaR cut
		#data = data[data.dilepton_charge == -1]#Dirac samples only, i.e. opposite sign leptons
		#data = data[data.IsoMuTrigger_flag == True]#muon trigger applied for event

		iters.append(data)
		# add cuts

	return pd.concat(iters)

def working_point_cuts(df):

	#df = df[df.nselectedJets_nominal < 6]
	df = df[df.EventObservables_nominal_ht < 180.0]
	df = df[df.leadingLeptons_nominal_mtw < 80.0]
	df = df[df.EventObservables_nominal_met < 80.0]
	df = df[df.dilepton_mass < 80.0]

	return df

def get_working_point(df, label):

	sig_df = df[(label == 1)]
	bkg_df = df[(label == 0)]

	print("df has {} entries, of which S {} and B {}:".format(df.shape[0], sig_df.shape[0], bkg_df.shape[0]))

	sig_df_cuts = working_point_cuts(sig_df)
	bkg_df_cuts = working_point_cuts(bkg_df)

	sig_eff = sig_df_cuts.shape[0]/sig_df.shape[0]
	bkg_eff = bkg_df_cuts.shape[0]/bkg_df.shape[0]

	return sig_eff, bkg_eff

def change_feat(df, array_bdt):
    for feat in array_bdt:
    	if df[feat].dtypes == object:
    		#print(df[feat])
    		df[feat] = df[feat].map(lambda x: x[0])
    		#print(df[feat])

    return df

path = "/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/19Sep20_notagger"
path1 = "/vols/cms/jd918/LLP/CMSSW_10_2_18/src/"
modelPath = os.path.join(path1,"bdt/bdt.model")
#modelPath = os.path.join(path1,"PhysicsTools/NanoAODTools/data/bdt/bdt.model")

array_preselection = [
			"nleadingLeptons", "nsubleadingLeptons", "nselectedJets_nominal",
			"dilepton_charge", "IsoMuTrigger_flag", "lepJet_nominal_deltaR",
			"MET_filter"
			]

with open(os.path.join(path1,"PhysicsTools/NanoAODTools/data/bdt/bdt_inputs.txt")) as f:
	array_bdt = [line.rstrip() for line in f]

array_list = array_preselection + array_bdt

n_events=300000

samples = {
			'HNL_dirac_all_ctau1p0e01_massHNL8p0_Vall9p475e-04-2016': [10, 8, 'orange'],
			'HNL_dirac_all_ctau1p0e02_massHNL2p0_Vall1p286e-02-2016': [100, 2, 'red'],
			'HNL_dirac_all_ctau1p0e-01_massHNL12p0_Vall3p272e-03-2016': [0.1, 12, 'green']
			}

useOneFile = False

if useOneFile:
	sig_df_dict = {sample: get_df(os.path.join(path,"2016",sample,"nano_1_Friend.root"), array_list, True) for sample in samples.keys()}
	sig_label_dict = {sample: np.ones(sig_df_dict[sample].shape[0]) for sample in samples}

	wjets_df = get_df(os.path.join(path,"2016/WToLNu_0J_13TeV-amcatnloFXFX-pythia8-ext1-2016/nano_1_Friend.root"), array_list, False)
	tt_df = get_df(os.path.join(path,"2016/TTToSemiLeptonic_*/nano_1_Friend.root"), array_list, False)
	dyjets_df = get_df(os.path.join(path,"2016/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-ext1-2016/nano_1_Friend.root"), array_list, False)

else:
	sig_df_dict = {sample: get_df(os.path.join(path,"2016",sample,"nano_*_Friend.root"), array_list, True) for sample in samples.keys()}
	sig_label_dict = {sample: np.ones(sig_df_dict[sample].shape[0]) for sample in samples}

	wjets_df = get_df(os.path.join(path,"2016/WToLNu_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	tt_df = get_df(os.path.join(path,"2016/TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	dyjets_df = get_df(os.path.join(path,"2016/DYJetsToLL*amcatnlo*/nano_*_Friend.root"), array_list, False).sample(n=n_events)


bkg_df = pd.concat([wjets_df, dyjets_df, tt_df])
bkg_label = np.zeros(bkg_df.shape[0])

df_dict = {sample: pd.concat([sig_df[array_bdt], bkg_df[array_bdt]]) for sample, sig_df in sig_df_dict.items()}
label_dict = {sample: np.concatenate([sig_label, bkg_label]) for sample, sig_label in sig_label_dict.items()}

for sample, df in df_dict.items():
	print(sample)
	print(df.shape[0])
	print(list(df))
	df = change_feat(df, array_bdt)
	df_dict[sample] = df.reindex(sorted(df.columns), axis=1)
	print(df)
	print(df_dict[sample])

working_points = {sample: get_working_point(df, label_dict[sample]) for sample, df in df_dict.items()}
print(working_points)

#df_dict = {sample: df[array_bdt] for sample, df in df_dict.items()}

model = XGBClassifier()
booster = Booster()
#model._le = LabelEncoder().fit([1])
booster.load_model(modelPath)
booster.feature_names = sorted(array_bdt)
model._Booster = booster
bdt_score = {}
for sample, df in df_dict.items():
	bdt_score[sample] = model.predict_proba(df)

fpr = {}
tpr = {}
for sample, df in df_dict.items():
	fpr[sample], tpr[sample], _ = sklearn.metrics.roc_curve(label_dict[sample], bdt_score[sample][:, 1])


#figx = pickle.load(open('BDT_roc.fig.pickle', 'rb'))
figx = pickle.load(open(os.path.join(path1,'bdt/BDT_roc.fig.pickle'), 'rb'))
figx.show()

bdt_roc = figx.axes[0].lines[0].get_data()
#bdt_roc_fpr = figx.axes[1].lines[0].get_data()
print(bdt_roc)
#print(bdt_roc_fpr)

fig, ax = pyplot.subplots()
ax.plot(bdt_roc[0], bdt_roc[1], label='all training data')
for sample, df in df_dict.items():
	print(samples[sample])
	print(samples[sample][2])
	print(working_points[sample])
	ax.plot(tpr[sample], fpr[sample], label=r'c$\tau_{0}$='+str(samples[sample][0])+'mm; m$_{HNL}$='+str(samples[sample][1])+'GeV', color=samples[sample][2])
	ax.plot(working_points[sample][0], working_points[sample][1], marker='x', markersize=5, color=samples[sample][2])

ax.legend(loc='lower right')
pyplot.yscale('log')
pyplot.xlabel('Signal Efficiency')
pyplot.ylabel('Background Efficiency')
pyplot.title('XGBoost ROC curve')
#pyplot.show()
pyplot.savefig(os.path.join(path1,'bdt/BDT_roc_samples.pdf'))
pyplot.savefig(os.path.join(path1,'bdt/BDT_roc_samples.png'))
