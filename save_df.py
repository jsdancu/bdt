# Script reading in all HNL and background MC ntuples, apply pre-selection cuts, calculate event weights,
# save relevant event features and event label (sig vs bkg) for BDT training and analysis

# j.dancu18@imperial.ac.uk, 2021

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
import json
import seaborn as sns
import argparse
import time
from datetime import date
from pathlib import Path
from termcolor import colored, cprint

import memory_usage

today = date.today()

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--luminosity', dest='luminosity', action='store',type=float, default=35.92)
parser.add_argument('--oneFile', dest='oneFile', action='store',type=int, default=1)
parser.add_argument('--bdt_df', dest='bdt_df', action='store',type=int, default=1)
parser.add_argument('--dir_bkg', dest='dir_bkg', action='store',default="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/03Dec20")
parser.add_argument('--dir_sig', dest='dir_sig', action='store',default="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/03Dec20_bdt")
parser.add_argument('--dir_src', dest='dir_src', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--bdt_inputs', dest='bdt_inputs', action='store',default="bdt/bdt_inputs_all.txt")
parser.add_argument('--dir_LLP', dest='dir_LLP', action='store',default="/vols/cms/LLP/")
parser.add_argument('--sm_xsec', dest='sm_xsec', action='store',default="xsec.json")
parser.add_argument('--hnl_xsec', dest='hnl_xsec', action='store',default="gridpackLookupTable.json")
parser.add_argument('--dir_yields', dest='dir_yields', action='store',default="yields_201117")
parser.add_argument('--file_yields', dest='file_yields', action='store',default="eventyields.json")
parser.add_argument('--file_hnl_yields', dest='file_hnl_yields', action='store',default="eventyieldsHNL.json")

args = parser.parse_args()

####################### Returns HNL coupling value for full muon-muon & electron-electron couling; half coupling for muon-electron/electron-muon mixing
def get_coupling(df, coupling):

	if coupling == 12:
		cut = (df["leadingLeptons_isElectron"] == 0) & (df["subleadingLeptons_isElectron"] == 0)
	elif coupling == 7:
		cut = ((df["leadingLeptons_isElectron"] == 0) & (df["subleadingLeptons_isElectron"] == 1)) | ((df["leadingLeptons_isElectron"] == 1) & (df["subleadingLeptons_isElectron"] == 1))
	elif coupling == 2:
		cut = (df["leadingLeptons_isElectron"] == 1) & (df["subleadingLeptons_isElectron"] == 0)

	return cut

######################### Calculates overall event weight for HNL and background processes ##################
def calc_event_weight(df, process, xsecs_raw, yields_raw, sig):

	if sig:
		coupling = 1
		#if HNL coupling is 1
		xsec_ = xsecs_hnl_raw[process.replace("-"+str(args.year), "")]["weights"][str(coupling)]["xsec"]["nominal"]
		for yield_key in yields_raw.keys():
			if yield_key in process:
				yield_ = yields_raw[yield_key]["weighted"]
				break
		#if HNL coupling not 1
		#xsec_ = df["LHEWeights_coupling_"+str(coupling)] * xsecs_hnl_raw[process.replace("-"+str(args.year), "")]["weights"][str(coupling)]["xsec"]["nominal"]
		#yield_ = yields_hnl_raw[process]["LHEWeights_coupling_"+str(coupling)]
		
		'''
		xsec_dict = {}
		yield_dict = {}
		for coupling in [2, 7, 12]:
			#coupling = get_coupling(df)
			xsec_dict[str(coupling)] = df["LHEWeights_coupling_"+str(coupling)] * xsecs_hnl_raw[process.replace("-"+str(args.year), "")]["weights"][str(coupling)]["xsec"]["nominal"]
			yield_dict[str(coupling)] = yields_hnl_raw[process]["LHEWeights_coupling_"+str(coupling)]
		'''
	else:
		for xsec_key in xsecs_raw.keys():
			if xsec_key in process:
				xsec_ = xsecs_raw[xsec_key]
				break
		for yield_key in yields_raw.keys():
			if yield_key in process:
				yield_ = yields_raw[yield_key]["weighted"]
				break

	print(f'x-sec: {xsec_}')
	print(f'yield: {yield_}')

	#get genweight
	genweight = df["genweight"]
	print(f'Mean genweight: {np.mean(genweight)} +/- {np.std(genweight)}')
	#cprint(f'NAN in genweight: {genweight.isna().any(axis=None)}', color='red')
	assert not genweight.isna().any(axis=None)

	#get pile-up weight
	pu_weight = df["puweight_nominal"]
	print(f'Mean pu_weight: {np.mean(pu_weight)} +/- {np.std(pu_weight)}')
	#cprint(f'NAN in pu_weight: {pu_weight.isna().any(axis=None)}', color='red')
	assert not pu_weight.isna().any(axis=None)

	#get ID & isolation weight
	id_iso_weight = df["tightMuons_weight_iso_nominal"] * df["tightMuons_weight_id_nominal"] * df["tightElectrons_weight_id_nominal"] * df["tightElectrons_weight_reco_nominal"] * df["looseElectrons_weight_reco_nominal"]
	print(f'Mean id_iso_weight: {np.mean(id_iso_weight)} +/- {np.std(id_iso_weight)}')
	#cprint(f'NAN in id_iso_weight: {id_iso_weight.isna().any(axis=None)}', color='red')
	assert not id_iso_weight.isna().any(axis=None)

	#get trigger weight
	trigger_weight = df["IsoMuTrigger_weight_trigger_nominal"]
	print(f'Mean trigger_weight: {np.mean(trigger_weight)} +/- {np.std(trigger_weight)}')
	#cprint(f'NAN in trigger_weight: {trigger_weight.isna().any(axis=None)}', color='red')
	assert not trigger_weight.isna().any(axis=None)

	if sig:
		'''
		for coupling in [2, 7, 12]:
			weight = eval(str(xsec_dict[str(coupling)])) * args.luminosity * 1e3 / yield_dict[str(coupling)]
			df["event_weight_"+str(coupling)] = df["LHEWeights_coupling_"+str(coupling)] * weight * genweight * pu_weight * id_iso_weight * trigger_weight
		
		for coupling in [2, 7, 12]:
			cut = get_coupling(df, coupling).astype(float)
			weight += cut * xsec_dict[str(coupling)] * args.luminosity * 1e3 / yield_dict[str(coupling)]
		
		'''
		#calculate weight from x-sec, lumi and yield
		weight = xsec_ * args.luminosity * 1e3 / yield_			
		event_weight = weight * genweight * pu_weight * id_iso_weight * trigger_weight

	else:
		#calculate weight from x-sec, lumi and yield
		weight = eval(str(xsec_)) * args.luminosity * 1e3 / yield_
		event_weight = weight * genweight * pu_weight * id_iso_weight * trigger_weight

	cprint(f'NAN in event_weight: {event_weight.isna().any(axis=None)}', color='red')
	df["event_weight"] = event_weight

############################## Applies pre-selection cuts on given process and returns as a dataframe ###########################
def get_df(path, processes, root_files, branches, sig, qcd, xsecs_raw, yields_raw):
	iters = []

	p = Path(path)
	processes_path = p.glob(processes)

	for process_path in processes_path:

		files = os.path.join(process_path, root_files)

		process = process_path.as_posix().split("/")[9]
		print(process)

		for data in uproot.pandas.iterate(files, "Friends", branches=branches, flatten=False):
			#print(path)

			data = data[data.nleadingLeptons == 1]#one leading lepton in event
			data = data[data.nsubleadingLeptons == 1]#one subleading lepton in event
			data = data[data.nselectedJets_nominal > 0]#at least one jet in event
			data = data[data.nselectedJets_nominal < 5]
			data = data[data.dilepton_mass < 80.]#apply CR cut
			data = data[data.dilepton_mass > 20.]#apply CR cut
			data = data[data.EventObservables_nominal_met < 100.]#apply CR cut
			data = data[data.MET_filter == 1]#apply MET filter

			data = data[data.category_simplified_nominal_index > 0]

			if  not qcd:
				data = data[((data.leadingLeptons_isElectron == 1) & (data.IsoElectronTrigger_flag == 1)) | ((data.leadingLeptons_isElectron == 0) & (data.IsoMuTrigger_flag == 1))]

			calc_event_weight(data, process, xsecs_raw, yields_raw, sig)
			iters.append(data)

	df = pd.concat(iters)
	print(f'event weight: {df["event_weight"]}')
	return df

'''
def get_df(path, processes, root_files, branches, sig, xsecs_raw, yields_raw):
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

	return pd.concat(iters)
'''

path_bkg = args.dir_bkg
path_sig = args.dir_sig
path1 = args.dir_src

################### Defines features/branches to be read in from ntuples for pre-selection cuts & event weight calculation ##########
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
array_HNL_preselection = ["LHEWeights_coupling_1", "LHEWeights_coupling_2", "LHEWeights_coupling_7", "LHEWeights_coupling_12"]

################## Reads in extensive BDT input feature set #########################
with open(os.path.join(path1, args.bdt_inputs)) as f:
	array_bdt = [line.rstrip() for line in f]

array_list = array_preselection + array_bdt

################## Reads in cross-section ratios for background samples ################
bkg_ratios = json.load(open(os.path.join(path1, "bdt/bkg_ratios_"+str(args.year)+".json")))
print(bkg_ratios)

if args.year == "2016":
	wjets_process = "WToLNu_*"
else:
	wjets_process = "WJetsToLNu_*"

################# Reads in cross-section and yield values for various processes (HNL & bkg MC) #############
xsecs_raw = json.load(open(os.path.join(args.dir_LLP, args.sm_xsec)))
xsecs_hnl_raw = json.load(open(os.path.join(args.dir_LLP, args.hnl_xsec)))

yields_raw = json.load(open(os.path.join(args.dir_LLP, args.dir_yields, args.year, args.file_yields)))
#if HNL coupling is 1
yields_hnl_raw = json.load(open(os.path.join(args.dir_LLP, args.dir_yields, args.year, args.file_yields)))
#if HNL coupling is not 1
#yields_hnl_raw = json.load(open(os.path.join(args.dir_LLP, args.dir_yields, args.year, args.file_hnl_yields)))

####################### Reads in data from ntuples per process (HNL or bkg MC), returns dataframes for BDT training or analysis
useOneFile = args.oneFile

if args.bdt_df:
	if useOneFile:
		sig_df = get_df(os.path.join(path_sig, str(args.year)), "HNL_majorana_all_ctau1p0e03_massHNL1p0_Vall1p668e-02-"+str(args.year), "nano_1_Friend.root", array_list, True, False, xsecs_hnl_raw, yields_hnl_raw)
		wjets_df = get_df(os.path.join(path_bkg, str(args.year)), wjets_process+"0J_*","nano_1_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
		vgamma_df = get_df(os.path.join(path_bkg, str(args.year)), "ZGToLLG_01J_*","nano_1_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
		qcd_df = get_df(os.path.join(path_sig, str(args.year)), "QCD_Pt-1000toInf_*","nano_1_Friend.root", array_list, False, True, xsecs_raw, yields_raw)
		dyjets_df = get_df(os.path.join(path_bkg, str(args.year)), "DYJetsToLL_M-10to50_*","nano_1_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
	else:
		sig_df = get_df(os.path.join(path_sig, str(args.year)), "HNL_majorana_*","nano_[1-4]_Friend.root", array_list, True, False, xsecs_hnl_raw, yields_hnl_raw)
		print("HNL: "+str(sig_df.shape[0]))
		#qcd_df = get_df(os.path.join(path_sig, str(args.year), "QCD_Pt-*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["qcd"]*sig_df.shape[0]))
		qcd_df = get_df(os.path.join(path_sig, str(args.year)), "QCD_Pt-*","nano_*_Friend.root", array_list, False, True, xsecs_raw, yields_raw).sample(n=int(bkg_ratios["qcd"]*sig_df.shape[0]))
		print("qcd: "+str(qcd_df.shape[0]))
		wjets_df = get_df(os.path.join(path_bkg, str(args.year)), wjets_process,"nano_*_Friend.root", array_list, False, False, xsecs_raw, yields_raw).sample(n=int(bkg_ratios["wjets"]*sig_df.shape[0]))
		print("wjets: "+str(wjets_df.shape[0]))
		vgamma_df = get_df(os.path.join(path_bkg, str(args.year)), "[WZ]GTo*","nano_*_Friend.root", array_list, False, False, xsecs_raw, yields_raw).sample(n=int(bkg_ratios["vgamma"]*sig_df.shape[0]))
		print("vgamma: "+str(vgamma_df.shape[0]))
		dyjets_df = get_df(os.path.join(path_bkg, str(args.year)), "DYJetsToLL*amcatnlo*","nano_*_Friend.root", array_list, False, False, xsecs_raw, yields_raw).sample(n=int(bkg_ratios["dyjets"]*sig_df.shape[0]))
		print("dyjets: "+str(dyjets_df.shape[0]))
		#tt_df = get_df(os.path.join(path, str(args.year), "TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)

	print("HNL: "+str(sig_df.shape[0]))
	print("wjets: "+str(wjets_df.shape[0]))
	print("vgamma: "+str(vgamma_df.shape[0]))
	print("dyjets: "+str(dyjets_df.shape[0]))
	print("qcd: "+str(qcd_df.shape[0]))

	bkg_df = pd.concat([wjets_df, vgamma_df, dyjets_df, qcd_df])
	print("total bkg: "+str(bkg_df.shape[0]))

	sig_label = np.ones(sig_df.shape[0])
	bkg_label = np.zeros(bkg_df.shape[0])

	wjets_label = 2*np.ones(wjets_df.shape[0])
	vgamma_label = 3*np.ones(vgamma_df.shape[0])
	dyjets_label = 4*np.ones(dyjets_df.shape[0])
	qcd_label = 5*np.ones(qcd_df.shape[0])

	print("sig_df")
	print(list(sig_df))

	print("bkg_df")
	print(list(bkg_df))

	################# Only save relevant features and concatenate signal & bkg final dataframes ########################
	extra_vars = ["genweight", "category_simplified_nominal_llpdnnx_m_llj", "category_simplified_nominal_index", "event_weight"]
	df = pd.concat([sig_df[array_bdt+extra_vars], bkg_df[array_bdt+extra_vars]])
	label = np.concatenate([sig_label, bkg_label])
	label_type = np.concatenate([sig_label, wjets_label, vgamma_label, dyjets_label, qcd_label])

	print(f'df: {df}')

	################# Save subleading jets in separate column ###############
	df["subselectedJets_nominal_pt"] = df["selectedJets_nominal_pt"].map(lambda x: x[1] if len(x)>1 else -1.)
	df["subselectedJets_nominal_eta"] = df["selectedJets_nominal_eta"].map(lambda x: x[1] if len(x)>1 else -1.)

	################# Features that are saved as array variables are converted to scalar ##############
	################# deltaPhi and eta variables are saved as absolute values ####################
	for feat in array_bdt:
		if df[feat].dtypes == object:
			df[feat] = df[feat].map(lambda x: x[0])
			if "deltaPhi" in feat:
				df[feat] = df[feat].map(lambda x: np.abs(x))
			if "eta" in feat:
				df[feat] = df[feat].map(lambda x: np.abs(x))

	################# Save process label in dataframe (signal or bkg) ########################
	df["label"] = label
	df["label_type"] = label_type

	print(df)

	##################### Save dataframe in pickle and csv files ######################
	if useOneFile:
		df.to_pickle(os.path.join(path1, "bdt/bdt_df_"+str(args.year)+"_smallie.pkl"))
		df.to_csv(os.path.join(path1, "bdt/bdt_df_"+str(args.year)+"_smallie.csv"))
	else:	
		df.to_pickle(os.path.join(path1, "bdt/bdt_df_"+str(args.year)+".pkl"))
		df.to_csv(os.path.join(path1, "bdt/bdt_df_"+str(args.year)+".csv"))

else:
	bkg_df = {}
	#################### Read in data from ntuples, save background dataframes in dictionary (for memory reasons) ##############
	if useOneFile:
		print(path_bkg)
		sig_df = get_df(os.path.join(path_bkg, str(args.year)), "HNL_majorana_all_ctau1p0e03_massHNL1p0_Vall1p668e-02-"+str(args.year),"nano_5_Friend.root", array_list, True, False, xsecs_hnl_raw, yields_hnl_raw)
		bkg_df["wjets_df"] = get_df(os.path.join(path_bkg, str(args.year)), wjets_process, "nano_1_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
		bkg_df["vgamma_df"] = get_df(os.path.join(path_bkg, str(args.year)), "ZGToLLG_01J_*","nano_1_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
		bkg_df["qcd_df"] = get_df(os.path.join(path_sig, str(args.year)), "QCD_Pt-1000toInf_*","nano_1_Friend.root", array_list, False, True, xsecs_raw, yields_raw)
		bkg_df["dyjets_df"] = get_df(os.path.join(path_bkg, str(args.year)), "DYJetsToLL_M-10to50_*","nano_1_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
		'''
		#tt_df = get_df(os.path.join(path, args.year, "TTToSemiLeptonic_*/nano_1_Friend.root"), array_list, False)
		'''
	else:
		print(path_bkg)
		sig_df = get_df(os.path.join(path_bkg, str(args.year)), "HNL_majorana_*","nano_*_Friend.root", array_list, True, False, xsecs_hnl_raw, yields_hnl_raw)
		print("HNL: "+str(sig_df.shape[0]))
		bkg_df["qcd_df"] = get_df(os.path.join(path_sig, str(args.year)), "QCD_Pt-*","nano_*_Friend.root", array_list, False, True, xsecs_raw, yields_raw)
		bkg_df["wjets_df"] = get_df(os.path.join(path_bkg, str(args.year)), wjets_process,"nano_*_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
		bkg_df["vgamma_df"] = get_df(os.path.join(path_bkg, str(args.year)), "[WZ]GTo*","nano_*_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
		bkg_df["dyjets_df"] = get_df(os.path.join(path_bkg, str(args.year)), "DYJetsToLL*amcatnlo*","nano_*_Friend.root", array_list, False, False, xsecs_raw, yields_raw)
		'''
		#tt_df = get_df(os.path.join(path, str(args.year), "TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
		'''

	print("HNL: "+str(sig_df.shape[0]))
	print("wjets: "+str(bkg_df["wjets_df"].shape[0]))
	print("vgamma: "+str(bkg_df["vgamma_df"].shape[0]))
	print("dyjets: "+str(bkg_df["dyjets_df"].shape[0]))
	print("qcd: "+str(bkg_df["qcd_df"].shape[0]))

	sig_label = np.ones(sig_df.shape[0])
	
	bkg_label = {}
	for key in bkg_df.keys():
		bkg_label[key] = np.zeros(bkg_df[key].shape[0])

	bkg_types = {
				"wjets_df": 2,
				"vgamma_df": 3,
				"dyjets_df": 4,
				"qcd_df": 5
				}
	bkg_label_type = {}
	for key in bkg_df.keys():
		bkg_label_type[key] = bkg_types[key]*np.ones(bkg_df[key].shape[0])

	print("sig_df")
	print(list(sig_df))

	################# Only save relevant features and concatenate signal & bkg final dataframes ########################
	extra_vars = ["genweight", "category_simplified_nominal_llpdnnx_m_llj", "category_simplified_nominal_index", "event_weight"]
	sig_df = sig_df[array_bdt+extra_vars]

	################# Save subleading jets in separate column ###############
	sig_df["subselectedJets_nominal_pt"] = sig_df["selectedJets_nominal_pt"].map(lambda x: x[1] if len(x)>1 else -1.)
	sig_df["subselectedJets_nominal_eta"] = sig_df["selectedJets_nominal_eta"].map(lambda x: x[1] if len(x)>1 else -1.)
	sig_df["label"] = sig_label
	sig_df["label_type"] = sig_label
	for key in bkg_df.keys():
		bkg_df[key] = bkg_df[key][array_bdt+extra_vars]
		bkg_df[key]["subselectedJets_nominal_pt"] = bkg_df[key]["selectedJets_nominal_pt"].map(lambda x: x[1] if len(x)>1 else -1.)
		bkg_df[key]["subselectedJets_nominal_eta"] = bkg_df[key]["selectedJets_nominal_eta"].map(lambda x: x[1] if len(x)>1 else -1.)
		bkg_df[key]["label"] = bkg_label[key]
		bkg_df[key]["label_type"] = bkg_label_type[key]
	'''
	df = pd.concat([sig_df[array_bdt+extra_vars], bkg_df[array_bdt+extra_vars]])
	label = np.concatenate([sig_label, bkg_label])
	label_type = np.concatenate([sig_label, wjets_label, vgamma_label, dyjets_label, qcd_label])
	'''

	################# Features that are saved as array variables are converted to scalar ##############
	################# deltaPhi and eta variables are saved as absolute values ####################
	################# Save process label in dataframe (signal or bkg) ########################
	for feat in array_bdt:
		if sig_df[feat].dtypes == object:
			sig_df[feat] = sig_df[feat].map(lambda x: x[0])
			for key in bkg_df.keys():
				bkg_df[key][feat] = bkg_df[key][feat].map(lambda x: x[0])
			if "deltaPhi" in feat:
				sig_df[feat] = sig_df[feat].map(lambda x: np.abs(x))
				for key in bkg_df.keys():
					bkg_df[key][feat] = bkg_df[key][feat].map(lambda x: np.abs(x))
			if "eta" in feat:
				sig_df[feat] = sig_df[feat].map(lambda x: np.abs(x))
				for key in bkg_df.keys():
					bkg_df[key][feat] = bkg_df[key][feat].map(lambda x: np.abs(x))

	##################### Concatenate all dataframes and hope it will not crash due to memory limits ###########
	bkg_df["all"] = pd.concat([bkg_df[key] for key in bkg_df.keys()])
	df = pd.concat([sig_df, bkg_df["all"]])
	print(f'df: {df}')
	print(f'df.event_weight: {df.event_weight}')

	##################### Save dataframe in pickle and csv files ######################
	if useOneFile:
		df.to_pickle(os.path.join(path1, "bdt/df_"+str(args.year)+"_smallie.pkl"))
		df.to_csv(os.path.join(path1, "bdt/df_"+str(args.year)+"_smallie.csv"))
		df["event_weight"].to_pickle(os.path.join(path1, "bdt/df_eventweight_"+str(args.year)+"_smallie.pkl"))
	else:	
		df.to_pickle(os.path.join(path1, "bdt/df_"+str(args.year)+".pkl"))
		df.to_csv(os.path.join(path1, "bdt/df_"+str(args.year)+".csv"))
		df["event_weight"].to_pickle(os.path.join(path1, "bdt/df_eventweight_"+str(args.year)+".pkl"))

####################### Print out memory usage (credits M. Mieskolainen) #####################
print(memory_usage.process_memory_use())
print(memory_usage.showmem())