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
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--oneFile', dest='oneFile', action='store',type=int, default=1)
parser.add_argument('--dir_data', dest='dir_data', action='store',default="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/19Sep20_notagger")
parser.add_argument('--dir_src', dest='dir_src', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--bdt_inputs', dest='bdt_inputs', action='store',default="PhysicsTools/NanoAODTools/data/bdt/bdt_inputs.txt")

args = parser.parse_args()

def get_df(path, branches, sig):
	iters = []
	for data in uproot.pandas.iterate(path, "Friends", branches=branches, flatten=False):
		#print(data.keys())

		data = data[data.nleadingLeptons == 1]#one leading lepton in event
		data = data[data.nsubleadingLeptons == 1]#one subleading lepton in event
		data = data[data.nselectedJets_nominal > 0]#at least one jet in event
		data = data[data.nselectedJets_nominal < 5]
		data = data[data.dilepton_mass < 85.]#apply CR cut
		data = data[data.dilepton_mass > 20.]#apply CR cut
		data = data[data.EventObservables_nominal_met < 100.]#apply CR cut
		data = data[data.MET_filter == 1]#apply MET filter

		data = data[data.selectedJets_nominal_minDeltaRSubtraction.map(lambda x: (x < 1.3).any())]#min DeltaR cut for signal region

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

def bdt_score_cut(df, bdt_score):

	df = df[bdt_score > 0.7]

	return df

def get_bdt_working_point(df, label, bdt_score):

	sig_df = df[(label == 1)]
	bkg_df = df[(label == 0)]

	print("df has {} entries, of which S {} and B {}:".format(df.shape[0], sig_df.shape[0], bkg_df.shape[0]))

	sig_df_cuts = bdt_score_cut(sig_df, bdt_score[(label == 1)])
	bkg_df_cuts = bdt_score_cut(bkg_df, bdt_score[(label == 0)])

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

path = args.dir_data
path1 = args.dir_src

modelPath = os.path.join(path1,"bdt/bdt.model")
#modelPath = os.path.join(path1,"PhysicsTools/NanoAODTools/data/bdt/bdt.model")

array_preselection = [
			"nleadingLeptons", "nsubleadingLeptons", "nselectedJets_nominal",
			"dilepton_charge", "IsoMuTrigger_flag", "lepJet_nominal_deltaR",
			"MET_filter", "selectedJets_nominal_minDeltaRSubtraction"
			]

with open(os.path.join(path1, args.bdt_inputs)) as f:
	array_bdt = [line.rstrip() for line in f]

array_list = array_preselection + array_bdt

n_events=30000

samples = {
			'HNL_majorana_all_ctau1p0e02_massHNL1p0_Vall5p274e-02-2016': [100, 1, 'orange'],
			'HNL_majorana_all_ctau1p0e01_massHNL6p0_Vall1p454e-03-2016': [10, 6, 'red'],
			'HNL_majorana_all_ctau1p0e-01_massHNL12p0_Vall2p314e-03-2016': [0.1, 12, 'green'],
			'HNL_majorana_*': [r'0.1-10$^{4}$', '1-20', 'blue']
			}

useOneFile = args.oneFile

if useOneFile:
	sig_df_dict = {sample: get_df(os.path.join(path,args.year,sample,"nano_1_Friend.root"), array_list, True) for sample in samples.keys()}
	sig_label_dict = {sample: np.ones(sig_df_dict[sample].shape[0]) for sample in samples}

	wjets_df = get_df(os.path.join(path, args.year, "WToLNu_0J_13TeV-amcatnloFXFX-pythia8-ext1-2016/nano_1_Friend.root"), array_list, False)
	tt_df = get_df(os.path.join(path, args.year, "TTToSemiLeptonic_*/nano_1_Friend.root"), array_list, False)
	dyjets_df = get_df(os.path.join(path, args.year, "DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-ext1-2016/nano_1_Friend.root"), array_list, False)

else:
	sig_df_dict = {sample: get_df(os.path.join(path, args.year ,sample,"nano_[5-20]_Friend.root"), array_list, True) for sample in samples.keys()}
	sig_label_dict = {sample: np.ones(sig_df_dict[sample].shape[0]) for sample in samples}

	wjets_df = get_df(os.path.join(path, args.year, "WToLNu_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	tt_df = get_df(os.path.join(path, args.year, "TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	dyjets_df = get_df(os.path.join(path, args.year, "DYJetsToLL*amcatnlo*/nano_*_Friend.root"), array_list, False).sample(n=n_events)


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

print('BDT cut performance:')

fpr = {}
tpr = {}
working_points_bdt_cut = {}
for sample, df in df_dict.items():
	fpr[sample], tpr[sample], _ = sklearn.metrics.roc_curve(label_dict[sample], bdt_score[sample][:, 1])
	working_points_bdt_cut[sample] = get_bdt_working_point(df, label_dict[sample], bdt_score[sample][:, 1])


#figx = pickle.load(open('BDT_roc.fig.pickle', 'rb'))
figx = pickle.load(open(os.path.join(path1,'bdt/BDT_roc.fig.pickle'), 'rb'))
figx.show()

bdt_roc = figx.axes[0].lines[0].get_data()
#bdt_roc_fpr = figx.axes[1].lines[0].get_data()
print(bdt_roc)
#print(bdt_roc_fpr)

fig, ax = pyplot.subplots()
ax.plot(bdt_roc[0], bdt_roc[1], label='all HNL samples')
for sample, df in df_dict.items():
	print(samples[sample])
	print(samples[sample][2])
	print(working_points[sample])
	if 'HNL_majorana_all_' in sample:
		ax.plot(tpr[sample], fpr[sample], label=r'c$\tau_{0}$='+str(samples[sample][0])+'mm; m$_{HNL}$='+str(samples[sample][1])+'GeV', color=samples[sample][2])
	ax.plot(working_points[sample][0], working_points[sample][1], marker='x', markersize=5, color=samples[sample][2])
	ax.plot(working_points_bdt_cut[sample][0], working_points_bdt_cut[sample][1], marker='o', markersize=5, color=samples[sample][2])

	print('x='+str(working_points_bdt_cut[sample][0]))
	print('y='+str(working_points_bdt_cut[sample][1]))
	ax.axhline(y=working_points_bdt_cut[sample][1], xmax=working_points_bdt_cut[sample][0], linestyle='--', color=samples[sample][2])
	ax.axvline(x=working_points_bdt_cut[sample][0], ymax=105.*working_points_bdt_cut[sample][1], linestyle='--', color=samples[sample][2])

ax.legend(title=r'x: cut-based; $\bullet$: BDT cut (0.7)', loc='lower right')
pyplot.yscale('log')
pyplot.xlabel('Signal Efficiency')
pyplot.ylabel('Background Efficiency')
pyplot.title('XGBoost ROC curve')
pyplot.ylim(1e-3, 1e0)
#pyplot.show()
pyplot.savefig(os.path.join(path1,'bdt/BDT_roc_samples.pdf'))
pyplot.savefig(os.path.join(path1,'bdt/BDT_roc_samples.png'))
