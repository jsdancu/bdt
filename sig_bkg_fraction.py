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
		#print(data.keys())

		data = data[data.nleadingLeptons == 1]#one leading lepton in event
		data = data[data.nsubleadingLeptons == 1]#one subleading lepton in event
		data = data[data.nselectedJets_nominal > 0]#at least one jet in event
		data = data[data.dilepton_mass < 80.]#apply CR cut
		data = data[data.dilepton_mass > 20.]#apply CR cut
		data = data[data.EventObservables_nominal_met < 100.]#apply CR cut
		data = data[data.MET_filter == 1]#apply MET filter

		#data = data[data.lepJet_nominal_deltaR < 2.0]#DeltaR cut

		iters.append(data)
		# add cuts

	return pd.concat(iters)

path = "/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/14Sep20_jet"
path1 = "/vols/cms/jd918/LLP/CMSSW_10_2_18/src/"

array_preselection = [
			"nleadingLeptons", "nsubleadingLeptons", "nselectedJets_nominal",
			"dilepton_charge", "IsoMuTrigger_flag", "lepJet_nominal_deltaR",
			"MET_filter"
			]

with open(os.path.join(path1, "PhysicsTools/NanoAODTools/data/bdt/bdt_inputs.txt")) as f:
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
	wjets_df = get_df(os.path.join(path,"2016/WToLNu_0J_13TeV-amcatnloFXFX-pythia8-ext1-2016/nano_1_Friend.root"), array_list, False)
	tt_df = get_df(os.path.join(path,"2016/TTToSemiLeptonic_*/nano_1_Friend.root"), array_list, False)
	dyjets_df = get_df(os.path.join(path,"2016/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-ext1-2016/nano_1_Friend.root"), array_list, False)
	#bkg_df = pd.concat([wjets_df, dyjets_df, tt_df]).sample(n=n_events)
else:
	sig_df_dict = {sample: get_df(os.path.join(path,"2016",sample,"nano_*_Friend.root"), array_list, True) for sample in samples.keys()}
	wjets_df = get_df(os.path.join(path,"2016/WToLNu_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	tt_df = get_df(os.path.join(path,"2016/TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	dyjets_df = get_df(os.path.join(path,"2016/DYJetsToLL*amcatnlo*/nano_*_Friend.root"), array_list, False).sample(n=n_events)

nsig_total = {sample:df.shape[0] for sample, df in sig_df_dict.items()}
print(nsig_total)

bkg_df = pd.concat([wjets_df, dyjets_df, tt_df])
nbkg_total = bkg_df.shape[0]
print(nbkg_total)

cuts = np.arange(0.0, 5.7, 0.1)
frac_total = {sample:float(nsig_total[sample])/float(nbkg_total)*np.ones(len(cuts)) for sample, df in sig_df_dict.items()}

frac1 = {sample:[] for sample in samples.keys()}
frac2 = {sample:[] for sample in samples.keys()}
frac = {sample:[] for sample in samples.keys()}

for cut in cuts:
	sig_df_dict_cut = {sample:df[df.lepJet_nominal_deltaR < cut] for sample, df in sig_df_dict.items()}
	bkg_df_cut = bkg_df[bkg_df.lepJet_nominal_deltaR < cut]

	for sample in samples.keys():
		if bkg_df_cut.shape[0] > 0:
			fraction1 = float(sig_df_dict_cut[sample].shape[0])/float(bkg_df_cut.shape[0])
		else:
			fraction1 = np.nan

		if (nbkg_total - bkg_df_cut.shape[0]) > 0:
			fraction2 = float(nsig_total[sample] - sig_df_dict_cut[sample].shape[0])/float(nbkg_total - bkg_df_cut.shape[0])
		else:
			fraction2 = np.nan

		#print(str(sample)+": cut: "+str(cut)+"frac: "+str(fraction))
		frac1[sample].append(fraction1)
		frac2[sample].append(fraction2)

		if fraction2 > 0.0:
			frac[sample].append(fraction1/fraction2)
		else:
			frac[sample].append(np.nan)

fig, ax = pyplot.subplots()
#ax.plot(cuts, frac_total, label=r'no $\Delta$R cut')
for sample in samples.keys():
	ax.plot(cuts, frac[sample], label=r'c$\tau_{0}$='+str(samples[sample][0])+'mm; m$_{HNL}$='+str(samples[sample][1])+'GeV', color=samples[sample][2])
	ax.plot(cuts, frac_total[sample], '--', label=r'no $\Delta$R cut', color=samples[sample][2])

ax.legend(loc='lower right')
pyplot.yscale('log')
pyplot.subplots_adjust(left=0.15)
pyplot.xlabel(r'Cut on $\Delta$R')
pyplot.ylabel('(TP/FP)/(FN/TN)')
pyplot.title(r'Signal/Background double fraction vs $\Delta$R cut')
#pyplot.show()
pyplot.savefig(os.path.join(path1,'bdt/sig_bkg_frac.pdf'))
pyplot.savefig(os.path.join(path1,'bdt/sig_bkg_frac.png'))
