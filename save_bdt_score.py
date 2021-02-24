import os
import uproot
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, Booster
from xgboost import plot_importance
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
from array import array

import cortools
import histogramming
import save_df_functions
import bdt_roc_curves

today = date.today()

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--oneFile', dest='oneFile', action='store',type=int, default=1)
#parser.add_argument('--dir_data', dest='dir_data', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/bdt/bdt_df_2016.pkl")
parser.add_argument('--dir_src', dest='dir_src', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--bdt_inputs', dest='bdt_inputs', action='store',default="bdt/bdt_inputs_hyperparam.json")
parser.add_argument('--combination', dest='combination', action='store',type=int, default=0)
parser.add_argument('--bdt_training_dir', dest='bdt_training_dir', action='store',default="bdt/bdt_training_hyperparam_20210222_2016_0")
parser.add_argument('--dir_sig', dest='dir_sig', action='store',default="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/12Feb21")
parser.add_argument('--dir_LLP', dest='dir_LLP', action='store',default="/vols/cms/LLP/")
parser.add_argument('--sm_xsec', dest='sm_xsec', action='store',default="xsec.json")
parser.add_argument('--hnl_xsec', dest='hnl_xsec', action='store',default="gridpackLookupTable.json")
parser.add_argument('--dir_yields', dest='dir_yields', action='store',default="yields_201117")
parser.add_argument('--file_yields', dest='file_yields', action='store',default="eventyields.json")
parser.add_argument('--file_hnl_yields', dest='file_hnl_yields', action='store',default="eventyieldsHNL.json")
args = parser.parse_args()

########################  Reading in the dataframe (analysis samples)  #########################
path = args.dir_src
path_sig = args.dir_sig

if args.oneFile:
	data = path+"bdt/df_"+str(args.year)+"_smallie.pkl"
else:
	data = path+"bdt/df_"+str(args.year)+".pkl"
bdt_inputs_file = path+args.bdt_inputs

bdt_training_dir = path+args.bdt_training_dir 
#bdt_training_dir = path+"bdt/bdt_training_hyperparam_"+str(today.strftime("%Y%m%d"))+"_"+str(args.year)+"_"+str(args.combination)
'''
if os.path.exists(bdt_training_dir):
	print("Overwriting dir")
else:
	os.mkdir(bdt_training_dir)
'''
df_biggie = pd.read_pickle(data)
print(df_biggie)
print(df_biggie["event_weight"])

array_bdt = json.load(open(bdt_inputs_file))[str(args.combination)]
print("array_bdt: ", array_bdt)

df = df_biggie[array_bdt]
df = df.reindex(sorted(df.columns), axis=1)
print(df)

label = df_biggie["label"]

df_bkg = df[(label == 0)]

#################################   Reading in individual HNL model files  ####################################
samples = {
			'HNL_majorana_all_ctau1p0e02_massHNL1p0_Vall5p274e-02-': [100, 1, 'orange'],
			'HNL_majorana_all_ctau1p0e01_massHNL6p0_Vall1p454e-03-': [10, 6, 'red'],
			'HNL_majorana_all_ctau1p0e-01_massHNL12p0_Vall2p314e-03-': [0.1, 12, 'green']
			}

array_preselection = [
			"nleadingLeptons", "nsubleadingLeptons", "nselectedJets_nominal",
			"dilepton_charge", "category_simplified_nominal_index", "dilepton_mass",
			"MET_filter", "selectedJets_nominal_minDeltaRSubtraction", "genweight",
            "category_simplified_nominal_llpdnnx_m_llj",
			"IsoMuTrigger_flag", "IsoElectronTrigger_flag",
			"tightMuons_weight_iso_nominal", "tightMuons_weight_id_nominal", 
			"tightElectrons_weight_id_nominal", "tightElectrons_weight_reco_nominal",
			"looseElectrons_weight_reco_nominal", "IsoMuTrigger_weight_trigger_nominal",
			"IsoMuTrigger_flag", "IsoElectronTrigger_flag", "puweight_nominal"
			]


xsecs_hnl_raw = json.load(open(os.path.join(args.dir_LLP, args.hnl_xsec)))
yields_hnl_raw = json.load(open(os.path.join(args.dir_LLP, args.dir_yields, args.year, args.file_yields)))

if args.oneFile:
	sig_df_dict = {sample: save_df_functions.get_df(os.path.join(path_sig, args.year), sample+str(args.year), "nano_5_Friend.root", array_list, True, False, xsecs_hnl_raw, yields_hnl_raw)[array_bdt] for sample in samples.keys()}
	sig_label_dict = {sample: np.ones(sig_df_dict[sample].shape[0]) for sample in samples}
else:
	sig_df_dict = {sample: save_df_functions.get_df(os.path.join(path_sig, args.year), sample+str(args.year),"nano_[5-20]_Friend.root", array_list, True, False, xsecs_hnl_raw, yields_hnl_raw)[array_bdt] for sample in samples.keys()}
	sig_label_dict = {sample: np.ones(sig_df_dict[sample].shape[0]) for sample in samples}

for sample in samples:
	print(f'sig_df_dict[{sample}]: {sig_df_dict[sample]}')


#################################   Reading in BDT model  #####################################################
model = XGBClassifier()
booster = Booster()
#model._le = LabelEncoder().fit([1])
booster.load_model(os.path.join(bdt_training_dir, 'bdt_'+str(args.year)+'.model'))
booster.feature_names = sorted(array_bdt)
model._Booster = booster
'''
bdt_score = {}
for sample, df in df_dict.items():
	bdt_score[sample] = model.predict_proba(df)
'''
#################################   Evaluating BDT score on data sample  ##################################
bdt_score = model.predict_proba(df)[:, 1]
pickle_file_object = open(os.path.join(bdt_training_dir, "bdt_score_"+str(args.year)+".pkl"), "wb") 
pickle.dump(bdt_score, pickle_file_object)
pickle_file_object.close()

##################### Producing ROC curves for various HNL models with working point  ###########################

fpr = []
tpr = []
fpr, tpr, _ = sklearn.metrics.roc_curve(label, bdt_score[:, 1])


###################  Producing 2D heatmaps between BDT score and 3-body mass per process + category  #####################
categories=["muonmuon", "muonelectron", "electronelectron", "electronmuon"]
processes=["sig", "bkg"]

def category_cut(df, category):

	if category == "muonmuon":
		cut = (df["leadingLeptons_isElectron"] == 0) & (df["subleadingLeptons_isElectron"] == 0)
	elif category == "muonelectron":
		cut = (df["leadingLeptons_isElectron"] == 0) & (df["subleadingLeptons_isElectron"] == 1)
	elif category == "electronelectron":
		cut = (df["leadingLeptons_isElectron"] == 1) & (df["subleadingLeptons_isElectron"] == 1)
	elif category == "electronmuon":
		cut = (df["leadingLeptons_isElectron"] == 1) & (df["subleadingLeptons_isElectron"] == 0)

	return cut

binning_mass = [26, 0.0, 200.0]
binning_mass_mi = np.linspace(0.0, 200, 26)
hist_mass = {process: {category: histogramming.define_plots_1D(binning_mass, "m(l_{1}l_{2}j) (GeV)", "Entries", "mass variable "+category+process) for category in categories} for process in processes}
#hist_gw_mass = histogramming.define_plots_1D(binning_mass, "m(l_{1}l_{2}j) (GeV)", "Entries", "mass variable (pos gw)")

binning_bdt = [51, 0.0, 1.0]
binning_bdt_mi = np.linspace(0.0, 1.0, 51)
hist_bdt = {process: {category: histogramming.define_plots_1D(binning_bdt, "BDT score", "Entries", "BDT score bkg "+category+process) for category in categories} for process in processes}

hist_heatmap = {process: {category: histogramming.define_plots_2D(binning_mass, binning_bdt, "m(l_{1}l_{2}j) (GeV)", "BDT score", "heat map bkg "+category+process) for category in categories} for process in processes}

dict_categories = {process: {category:{} for category in categories} for process in processes}

for process in processes:
	for category in categories:

		cut = category_cut(df_biggie, category)
		print(category+"\n")
		
		if process=="bkg":
			invmass_category = df_biggie[cut & (label == 0)]["category_simplified_nominal_llpdnnx_m_llj"]
			bdt_score_category = bdt_score[cut & (label == 0)]
			weights = df_biggie[cut & (label == 0)]["event_weight"].to_numpy()
			weights[(df_biggie[cut & (label == 0)]["label_type"] == 5)] = 1.0
			#weights = np.ones(len(bdt_score_category))

		else:
			invmass_category = df_biggie[cut & (label == 1)]["category_simplified_nominal_llpdnnx_m_llj"]
			bdt_score_category = bdt_score[cut & (label == 1)]
			weights = np.ones(len(bdt_score_category))
			#print("sig: ", df_biggie[cut & (label == 1)]["event_weight"].to_numpy())
			#weights = df_biggie[cut & (label == 1)]["event_weight"].to_numpy()

		#defining weights -> should reevaluate this with new df!!!!
		#weights = np.ones(len(bdt_score_category))
		weights_bdt_score = np.ones(len(bdt_score_category))

		hist_mass[process][category] = histogramming.plotting_1D(invmass_category, hist_mass[process][category], weights)
		hist_bdt[process][category] = histogramming.plotting_1D(bdt_score_category, hist_bdt[process][category], weights_bdt_score)
		hist_heatmap[process][category] = histogramming.plotting_2D(invmass_category, bdt_score_category, hist_heatmap[process][category], weights)
		hist_heatmap[process][category].SetMinimum(0)

		########################  Calculate Pearson corr. coeff. & Mutual Information for process + category #######################
		dict_categories[process][category]["Pearson corr coeff"] = cortools.pearson_corr(invmass_category, bdt_score_category, weights)[0]
		dict_categories[process][category]["Mutual Information"] = cortools.mutual_information(invmass_category, bdt_score_category, weights)

		histogramming.plotting(hist_mass[process][category], bdt_training_dir, "massVar_heatmap_", dict_categories[process][category]["Mutual Information"], category, args.year, process)
		histogramming.plotting(hist_bdt[process][category], bdt_training_dir, "bdt_score_heatmap_", dict_categories[process][category]["Mutual Information"], category, args.year, process)
		histogramming.plotting(hist_heatmap[process][category], bdt_training_dir, "massVar_bdt_score_heatmap_", dict_categories[process][category]["Mutual Information"], category, args.year, process, drawstyle="COLZ")

#########################  Save Pearson corr. coeff & Mutual Information for all processes + categories in a json file ##############
with open(os.path.join(bdt_training_dir,'BDT_model_correlation.json'), 'w') as json_file_object:
    json.dump(dict_categories, json_file_object, indent=1)
