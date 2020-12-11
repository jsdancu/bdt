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
import scipy.interpolate
from matplotlib import pyplot
import pickle
import json
import seaborn as sns
import argparse
import time
from datetime import date

today = date.today()

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--oneFile', dest='oneFile', action='store',type=int, default=1)
parser.add_argument('--dir_data', dest='dir_data', action='store',default="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/03Dec20")
parser.add_argument('--dir_data_special', dest='dir_data_special', action='store',default="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/03Dec20_bdt")
#parser.add_argument('--dir_data_old', dest='dir_data', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/ntuples")
parser.add_argument('--dir_src', dest='dir_src', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--bdt_inputs', dest='bdt_inputs', action='store',default="PhysicsTools/NanoAODTools/data/bdt/bdt_inputs.txt")
parser.add_argument('--bdt_training_dir1', dest='bdt_training_dir1', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/bdt/bdt_training_20201204_2016")
parser.add_argument('--bdt_training_dir2', dest='bdt_training_dir2', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/bdt/bdt_training_20201210_2016")

args = parser.parse_args()

def get_df(path, branches, sig):
	iters = []
	for data in uproot.pandas.iterate(path, "Friends", branches=branches, flatten=False):
		#print(path)

		data = data[data.nleadingLeptons == 1]#one leading lepton in event
		data = data[data.nsubleadingLeptons == 1]#one subleading lepton in event
		data = data[data.nselectedJets_nominal > 0]#at least one jet in event
		data = data[data.nselectedJets_nominal < 5]
		data = data[data.dilepton_mass < 91.1876-15.]#apply CR cut
		data = data[data.dilepton_mass > 20.]#apply CR cut
		data = data[data.EventObservables_nominal_met < 100.]#apply CR cut
		data = data[data.MET_filter == 1]#apply MET filter

		data = data[data.category_simplified_nominal_index > 0]

		iters.append(data)
		# add cuts

	return pd.concat(iters)

def get_df_invmass(path, branches, sig):
	iters = []
	for data in uproot.pandas.iterate(path, "Friends", branches=branches, flatten=False):
		#print(path)

		data = data[data.nleadingLeptons == 1]#one leading lepton in event
		data = data[data.nsubleadingLeptons == 1]#one subleading lepton in event
		data = data[data.nselectedJets_nominal > 0]#at least one jet in event
		data = data[data.nselectedJets_nominal < 5]
		data = data[data.dilepton_mass < 91.1876-15.]#apply CR cut
		data = data[data.dilepton_mass > 20.]#apply CR cut
		data = data[data.EventObservables_nominal_met < 100.]#apply CR cut
		data = data[data.MET_filter == 1]#apply MET filter
		#adding 3-body invariant mass cut
		data = data[data.category_simplified_nominal_llpdnnx_m_llj < 90.]#apply CR cut
		data = data[data.category_simplified_nominal_llpdnnx_m_llj > 70.]#apply CR cut

		data = data[data.category_simplified_nominal_index > 0]#events in SR

		iters.append(data)
		# add cuts

	return pd.concat(iters)

def calc_roc(label, df):

	fpr = []
	tpr = []
	max_val = float(np.amax(df))
	step = max_val/1000.
	cut = 0.0
	n_sig = np.sum(label)
	n_bkg = len(label) - n_sig

	while cut < max_val:
		
		fpr_cut = np.sum(((df < cut).to_numpy().flatten()) & (np.array(label == 0)))/float(n_bkg)
		tpr_cut = np.sum(((df < cut).to_numpy().flatten()) & (np.array(label == 1)))/float(n_sig)
		fpr.append(fpr_cut)
		tpr.append(tpr_cut)
		cut += step

	return fpr, tpr

bdt_training_dir1 = args.bdt_training_dir1
bdt_training_dir2 = args.bdt_training_dir2

path = args.dir_data
path_special = args.dir_data_special
path1 = args.dir_src
path2 = os.path.join(args.dir_src, bdt_training_dir1)
path3 = os.path.join(args.dir_src, bdt_training_dir2)

array_preselection = [
			"nleadingLeptons", "nsubleadingLeptons", "nselectedJets_nominal",
			"dilepton_charge", "category_simplified_nominal_index", "dilepton_mass",
			"MET_filter", "selectedJets_nominal_minDeltaRSubtraction", "genweight",
            "EventObservables_nominal_met", "category_simplified_nominal_llpdnnx_m_llj"
			]

with open(os.path.join(path2, "bdt_inputs.txt")) as f:
	array_bdt_corr = [line.rstrip() for line in f]

with open(os.path.join(path3, "bdt_inputs.txt")) as f:
	array_bdt_uncorr = [line.rstrip() for line in f]

var_scan = ['EventObservables_nominal_ht']
array_list = array_preselection + array_bdt_corr

bkg_ratios = json.load(open(os.path.join(path1, "bdt/bkg_ratios_"+str(args.year)+".json")))
print(bkg_ratios)

if args.year == "2016":
	wjets_process = "WToLNu_"
else:
	wjets_process = "WJetsToLNu_"

useOneFile = args.oneFile

if useOneFile:
	sig_df = get_df(os.path.join(path, str(args.year), "HNL_majorana_all_ctau1p0e03_massHNL1p0_Vall1p668e-02-"+str(args.year)+"/nano_5_Friend.root"), array_list, True)
	wjets_df = get_df(os.path.join(path, str(args.year), wjets_process+"0J_*/nano_1_Friend.root"), array_list, False)
	vgamma_df = get_df(os.path.join(path, str(args.year), "ZGToLLG_01J_*/nano_1_Friend.root"), array_list, False)
	#qcd_df = get_df(os.path.join(path, str(args.year), "QCD_loose/QCD_Pt-1000toInf_*/nano_1_Friend.root"), array_list, False)
	qcd_df = get_df(os.path.join(path_special, str(args.year), "QCD_Pt-1000toInf_*/nano_1_Friend.root"), array_list, False)
	dyjets_df = get_df(os.path.join(path, str(args.year), "DYJetsToLL_M-10to50_*/nano_1_Friend.root"), array_list, False)
	#bkg_df = pd.concat([wjets_df, dyjets_df, tt_df]).sample(n=n_events)
else:
	#sig_df = get_df(os.path.join(path, str(args.year), "HNL_majorana_*/nano_[1-4]_Friend.root"), array_list, True)
	sig_df = get_df(os.path.join(path, str(args.year), "HNL_majorana_*/nano_*_Friend.root"), array_list, True)
	wjets_df = get_df(os.path.join(path, str(args.year), wjets_process+"*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["wjets"]*sig_df.shape[0]))
	vgamma_df = get_df(os.path.join(path, str(args.year), "[WZ]GTo*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["vgamma"]*sig_df.shape[0]))
	#qcd_df = get_df(os.path.join(path, str(args.year), "QCD_loose/QCD_Pt-*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["qcd"]*sig_df.shape[0]))
	qcd_df = get_df(os.path.join(path_special, str(args.year), "QCD_Pt-*/nano_*_Friend.root"), array_list, False)#.sample(n=int(bkg_ratios["qcd"]*sig_df.shape[0]))
	dyjets_df = get_df(os.path.join(path, str(args.year), "DYJetsToLL*amcatnlo*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["dyjets"]*sig_df.shape[0]))

print("HNL: "+str(sig_df.shape[0]))
print("wjets: "+str(wjets_df.shape[0]))
print("vgamma: "+str(vgamma_df.shape[0]))
print("dyjets: "+str(dyjets_df.shape[0]))
print("qcd: "+str(qcd_df.shape[0]))

bkg_df = pd.concat([wjets_df, vgamma_df, dyjets_df, qcd_df])
#bkg_df = pd.concat([wjets_df, vgamma_df, dyjets_df])
print("total bkg: "+str(bkg_df.shape[0]))

sig_label = np.ones(sig_df.shape[0])
bkg_label = np.zeros(bkg_df.shape[0])

print("sig_df")
print(list(sig_df))

print("bkg_df")
print(list(bkg_df))

df = pd.concat([sig_df, bkg_df])
label = np.concatenate([sig_label, bkg_label])

df["subselectedJets_nominal_pt"] = df["selectedJets_nominal_pt"].map(lambda x: x[1] if len(x)>1 else -1.)
df["subselectedJets_nominal_eta"] = df["selectedJets_nominal_eta"].map(lambda x: x[1] if len(x)>1 else -1.)

for feat in array_bdt_corr:
	if df[feat].dtypes == object:
		#print(df[feat])
		df[feat] = df[feat].map(lambda x: x[0])
		#print(df[feat])
		if "deltaPhi" in feat:
			df[feat] = df[feat].map(lambda x: np.abs(x))
		if "eta" in feat:
			df[feat] = df[feat].map(lambda x: np.abs(x))

array_bdt_corr += ["subselectedJets_nominal_pt"]
array_bdt_uncorr += ["subselectedJets_nominal_pt", "subselectedJets_nominal_eta"]

invmass_cut = (df["category_simplified_nominal_llpdnnx_m_llj"] < 90.) & (df["category_simplified_nominal_llpdnnx_m_llj"] > 70.)
df_invmass_cut = df[invmass_cut > 0]
label_invmass_cut = label[invmass_cut > 0]
print("df_invmass_cut: ", df_invmass_cut)
print("label_invmass_cut: ", label_invmass_cut)
print("len(df_invmass_cut): ", len(df_invmass_cut))
print("len(label_invmass_cut): ", len(label_invmass_cut))

#get HT scan ROC curve
print("HT scan ROC")
fpr_var_scan, tpr_var_scan = calc_roc(label, df[var_scan])

#reading in ROC curve from correlated BDT
figx = pickle.load(open(os.path.join(bdt_training_dir1,'BDT_roc_'+str(args.year)+'.fig.pickle'), 'rb'))
figx.show()

bdt_roc_corr = figx.axes[0].lines[0].get_data()
print("Correlated BDT ROC")
print(bdt_roc_corr)

#reading in ROC curve from uncorrelated BDT
figx = pickle.load(open(os.path.join(bdt_training_dir2,'BDT_roc_'+str(args.year)+'.fig.pickle'), 'rb'))
figx.show()

bdt_roc_uncorr = figx.axes[0].lines[0].get_data()
print("Uncorrelated BDT ROC")
print(bdt_roc_uncorr)

fpr_var_interpol = scipy.interpolate.interp1d(fpr_var_scan, tpr_var_scan)
fpr_var_interpol_1 = float(fpr_var_interpol(0.1))

fpr_corr_interpol = scipy.interpolate.interp1d(bdt_roc_corr[1], bdt_roc_corr[0])
fpr_corr_interpol_1 = float(fpr_corr_interpol(0.1))

fpr_uncorr_interpol = scipy.interpolate.interp1d(bdt_roc_uncorr[1], bdt_roc_uncorr[0])
fpr_uncorr_interpol_1 = float(fpr_uncorr_interpol(0.1))

colours = ['blue', 'orange', 'green', 'red']

fig, ax = pyplot.subplots()
ax.plot(tpr_var_scan, fpr_var_scan, label=r'$H_{T}$ scan', color=colours[0])
ax.plot(bdt_roc_corr[0], bdt_roc_corr[1], label='Correlated BDT', color=colours[1])
ax.plot(bdt_roc_uncorr[0], bdt_roc_uncorr[1], label='Uncorrelated BDT', color=colours[2])
pyplot.hlines(1e-1, 0., fpr_corr_interpol_1, linestyle="dashed", color='black')
pyplot.vlines(fpr_var_interpol_1, 1e-3, 1e-1, linestyle="dashed", color=colours[0])
pyplot.vlines(fpr_corr_interpol_1, 1e-3, 1e-1, linestyle="dashed", color=colours[1])
pyplot.vlines(fpr_uncorr_interpol_1, 1e-3, 1e-1, linestyle="dashed", color=colours[2])
pyplot.legend()
pyplot.yscale('log')
pyplot.xlabel('Signal Efficiency')
pyplot.ylabel('Background Efficiency')
pyplot.title('XGBoost ROC curve comparison - '+str(args.year))
pyplot.ylim(1e-3, 1e0)
pyplot.xlim(0., 1.)
#pyplot.show()
pyplot.savefig(os.path.join(path1,'bdt/BDT_roc_samples_var_scan_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path1,'bdt/BDT_roc_samples_var_scan_'+str(args.year)+'.png'))
pyplot.clf()

#Reevaluate ROC curves with 3-body invariant mass included
#HT scan with 3-body mass included
fpr_var_scan, tpr_var_scan = calc_roc(label_invmass_cut, df_invmass_cut[var_scan])

#read in correlated BDT model and evaluate it on the df 
model_corr = XGBClassifier()
booster = Booster()
booster.load_model(os.path.join(path2,'bdt_'+str(args.year)+'.model'))
booster.feature_names = array_bdt_corr
model_corr._Booster = booster

bdt_score_corr = model_corr.predict_proba(df_invmass_cut[array_bdt_corr])
fpr_corr, tpr_corr, _ = sklearn.metrics.roc_curve(label_invmass_cut, bdt_score_corr[:, 1])

#read in uncorrelated BDT model and evaluate it on the df 
model_uncorr = XGBClassifier()
booster = Booster()
booster.load_model(os.path.join(path3,'bdt_'+str(args.year)+'.model'))
booster.feature_names = array_bdt_uncorr
model_uncorr._Booster = booster

bdt_score_uncorr = model_uncorr.predict_proba(df_invmass_cut[array_bdt_uncorr])
fpr_uncorr, tpr_uncorr, _ = sklearn.metrics.roc_curve(label_invmass_cut, bdt_score_uncorr[:, 1])

fpr_var_interpol = scipy.interpolate.interp1d(fpr_var_scan, tpr_var_scan)
fpr_var_interpol_1 = float(fpr_var_interpol(0.1))

fpr_corr_interpol = scipy.interpolate.interp1d(fpr_corr, tpr_corr)
fpr_corr_interpol_1 = float(fpr_corr_interpol(0.1))

fpr_uncorr_interpol = scipy.interpolate.interp1d(fpr_uncorr, tpr_uncorr)
fpr_uncorr_interpol_1 = float(fpr_uncorr_interpol(0.1))

fig, ax = pyplot.subplots()
ax.plot(tpr_var_scan, fpr_var_scan, label=r'$H_{T}$ scan', color=colours[0])
ax.plot(tpr_corr, fpr_corr, label='Correlated BDT', color=colours[1])
ax.plot(tpr_uncorr, fpr_uncorr, label='Uncorrelated BDT', color=colours[2])
pyplot.hlines(1e-1, 0., fpr_corr_interpol_1, linestyle="dashed", color='black')
pyplot.vlines(fpr_var_interpol_1, 1e-3, 1e-1, linestyle="dashed", color=colours[0])
pyplot.vlines(fpr_corr_interpol_1, 1e-3, 1e-1, linestyle="dashed", color=colours[1])
pyplot.vlines(fpr_uncorr_interpol_1, 1e-3, 1e-1, linestyle="dashed", color=colours[2])
pyplot.legend(title=r"70 GeV<$m_{llj}$<90 GeV")
pyplot.yscale('log')
pyplot.xlabel('Signal Efficiency')
pyplot.ylabel('Background Efficiency')
pyplot.title('XGBoost ROC curve comparison - '+str(args.year))
pyplot.ylim(1e-3, 1e0)
pyplot.xlim(0., 1.)
#pyplot.show()
pyplot.savefig(os.path.join(path1,'bdt/BDT_roc_samples_var_scan_invmass_cut_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path1,'bdt/BDT_roc_samples_var_scan_invmass_cut_'+str(args.year)+'.png'))
pyplot.clf()