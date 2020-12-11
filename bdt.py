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

today = date.today()

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--oneFile', dest='oneFile', action='store',type=int, default=1)
parser.add_argument('--dir_data', dest='dir_data', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/ntuples")
parser.add_argument('--dir_src', dest='dir_src', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--bdt_inputs', dest='bdt_inputs', action='store',default="PhysicsTools/NanoAODTools/data/bdt/bdt_inputs.txt")

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
		#data = data[data.selectedJets_nominal_minDeltaRSubtraction.map(lambda x: (x < 1.3).any())]#min DeltaR cut for signal region

		#data = data[data.dilepton_charge == -1]#Dirac samples only, i.e. opposite sign leptons
		#data = data[data.IsoMuTrigger_flag == True]#muon trigger applied for event

		iters.append(data)
		# add cuts

	return pd.concat(iters)

bdt_training_dir = "bdt/bdt_training_"+str(today.strftime("%Y%m%d"))+"_"+str(args.year)

if os.path.exists(bdt_training_dir):
	print("Overwriting dir")
else:
	os.mkdir(bdt_training_dir)

path = args.dir_data
path1 = args.dir_src
path2 = os.path.join(args.dir_src, bdt_training_dir)

array_preselection = [
			"nleadingLeptons", "nsubleadingLeptons", "nselectedJets_nominal",
			"dilepton_charge", "category_simplified_nominal_index", "dilepton_mass",
			"MET_filter", "selectedJets_nominal_minDeltaRSubtraction", "genweight"
			]

with open(os.path.join(path1, args.bdt_inputs)) as f:
	array_bdt = [line.rstrip() for line in f]

array_list = array_preselection + array_bdt

bkg_ratios = json.load(open(os.path.join(path1, "bdt/bkg_ratios_"+str(args.year)+".json")))
print(bkg_ratios)

if args.year == "2016":
	wjets_process = "WToLNu_"
else:
	wjets_process = "WJetsToLNu_"

useOneFile = args.oneFile

if useOneFile:
	sig_df = get_df(os.path.join(path, str(args.year), "HNL_majorana_all_ctau1p0e03_massHNL1p0_Vall1p668e-02-"+str(args.year)+"/nano_1_Friend.root"), array_list, True)
	wjets_df = get_df(os.path.join(path, str(args.year), wjets_process+"0J_*/nano_1_Friend.root"), array_list, False)
	vgamma_df = get_df(os.path.join(path, str(args.year), "ZGToLLG_01J_*/nano_1_Friend.root"), array_list, False)
	#tt_df = get_df(os.path.join(path, args.year, "TTToSemiLeptonic_*/nano_1_Friend.root"), array_list, False)
	qcd_df = get_df(os.path.join(path, str(args.year), "QCD_loose/QCD_Pt-1000toInf_*/nano_1_Friend.root"), array_list, False)
	dyjets_df = get_df(os.path.join(path, str(args.year), "DYJetsToLL_M-10to50_*/nano_1_Friend.root"), array_list, False)
	#bkg_df = pd.concat([wjets_df, dyjets_df, tt_df]).sample(n=n_events)
else:
	'''
	sig_df = get_df(os.path.join(path, args.year, "HNL_majorana_*/nano_[1-4]_Friend.root"), array_list, True)
	wjets_df = get_df(os.path.join(path, args.year, "WToLNu_*/nano_*_Friend.root"), array_list, False).sample(n=int(0.442*sig_df.shape[0]))#(n=35360)
	vgamma_df = get_df(os.path.join(path, args.year, "[WZ]GTo*/nano_*_Friend.root"), array_list, False).sample(n=int(0.076*sig_df.shape[0]))#(n=6080)
	#tt_df = get_df(os.path.join(path, args.year, "TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	qcd_df = get_df(os.path.join(path, args.year, "QCD_loose/QCD_Pt-*/nano_*_Friend.root"), array_list, False).sample(n=int(0.272*sig_df.shape[0]))
	dyjets_df = get_df(os.path.join(path, args.year, "DYJetsToLL*amcatnlo*/nano_*_Friend.root"), array_list, False).sample(n=int(0.21*sig_df.shape[0]))#(n=16800)
	'''
	sig_df = get_df(os.path.join(path, str(args.year), "HNL_majorana_*/nano_[1-4]_Friend.root"), array_list, True)
	wjets_df = get_df(os.path.join(path, str(args.year), wjets_process+"*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["wjets"]*sig_df.shape[0]))
	vgamma_df = get_df(os.path.join(path, str(args.year), "[WZ]GTo*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["vgamma"]*sig_df.shape[0]))
	#tt_df = get_df(os.path.join(path, str(args.year), "TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
	qcd_df = get_df(os.path.join(path, str(args.year), "QCD_loose/QCD_Pt-*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["qcd"]*sig_df.shape[0]))
	dyjets_df = get_df(os.path.join(path, str(args.year), "DYJetsToLL*amcatnlo*/nano_*_Friend.root"), array_list, False).sample(n=int(bkg_ratios["dyjets"]*sig_df.shape[0]))

print("HNL: "+str(sig_df.shape[0]))
print("wjets: "+str(wjets_df.shape[0]))
print("vgamma: "+str(vgamma_df.shape[0]))
print("dyjets: "+str(dyjets_df.shape[0]))
print("qcd: "+str(qcd_df.shape[0]))

bkg_df = pd.concat([wjets_df, vgamma_df, dyjets_df, qcd_df])
print("total bkg: "+str(bkg_df.shape[0]))

sig_label = np.ones(sig_df.shape[0])
bkg_label = np.zeros(bkg_df.shape[0])

print("sig_df")
print(list(sig_df))

print("bkg_df")
print(list(bkg_df))


df = pd.concat([sig_df[array_bdt+["genweight"]], bkg_df[array_bdt+["genweight"]]])
label = np.concatenate([sig_label, bkg_label])

print(df)

df["subselectedJets_nominal_pt"] = df["selectedJets_nominal_pt"].map(lambda x: x[1] if len(x)>1 else -1.)
df["subselectedJets_nominal_eta"] = df["selectedJets_nominal_eta"].map(lambda x: x[1] if len(x)>1 else -1.)

for feat in array_bdt:
	if df[feat].dtypes == object:
		#print(df[feat])
		df[feat] = df[feat].map(lambda x: x[0])
		#print(df[feat])
		if "deltaPhi" in feat:
			df[feat] = df[feat].map(lambda x: np.abs(x))
		if "eta" in feat:
			df[feat] = df[feat].map(lambda x: np.abs(x))

array_bdt += ["subselectedJets_nominal_pt", "subselectedJets_nominal_eta"]

df = df.reindex(sorted(df.columns), axis=1)
print(df)

X_train1, X_test1, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=42, stratify=label)

X_train = X_train1[array_bdt]
X_test = X_test1[array_bdt]
df = df[array_bdt]

genweight = X_train1["genweight"]

# fit model no training data
model = XGBClassifier(objective='binary:logistic', learning_rate=0.05, max_depth=4, n_estimators=1000, nthread=-1)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, early_stopping_rounds=10, sample_weight=np.sign(genweight), verbose=True)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

log_loss = log_loss(y_test, y_pred)
print('log loss = {:.3f}'.format(log_loss))

# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

#model._Booster.save_model(os.path.join(path2,"icsTools/NanoAODTools/data/bdt/bdt.model"))
model._Booster.save_model(os.path.join(path2,'bdt_'+str(args.year)+'.model'))

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(path2,'BDT_loss_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path2,'BDT_loss_'+str(args.year)+'.png'))

# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(path2,'BDT_error_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path2,'BDT_error_'+str(args.year)+'.png'))

# take the second column because the classifier outputs scores for
# the 0 class as well
probs = model.predict_proba(X_test)[:, 1]

fig, axes = pyplot.subplots()
pyplot.hist(probs[y_test==0], label='background', bins = 50, histtype='step')
pyplot.hist(probs[y_test==1], label='signal', bins = 50, histtype='step')
axes.legend()
pyplot.ylabel('BDT output')
pyplot.xlabel('signal (1) vs background (0)')
pyplot.title('XGBoost separation - '+str(args.year))
pyplot.savefig(os.path.join(path2,'BDT_output_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path2,'BDT_output_'+str(args.year)+'.png'))

# fpr means false-positive-rate
# tpr means true-positive-rate
fpr, tpr, _ = metrics.roc_curve(y_test, probs)

#auc_score = 1.0 - metrics.auc(fpr, tpr)
auc_score = metrics.auc(tpr, fpr)
print('AUC = {:.3f}'.format(auc_score))#want it as small as possible

fig, ax = pyplot.subplots()
ax.plot(tpr, fpr, label='AUC = {:.3f}'.format(auc_score))
ax.legend(loc='lower right')
pyplot.yscale('log')
pyplot.xlabel('Signal Efficiency')
pyplot.ylabel('Background Efficiency')
pyplot.title('XGBoost ROC curve - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(path2,'BDT_roc_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path2,'BDT_roc_'+str(args.year)+'.png'))

pickle.dump(fig, open(os.path.join(path2,'BDT_roc_'+str(args.year)+'.fig.pickle'), 'wb'))

rename_dict = {
			'EventObservables_nominal_met': 'MET', 'EventObservables_nominal_ht': 'HT',
			'leadingLeptons_nominal_mtw': 'mT', 'leadingLeptons_nominal_deltaPhi': '|deltaPhi(l1, MET)|',
			'leadingLeptons_pt': 'pT(l1)', 'leadingLeptons_eta': '|eta(l1)|',
			'leadingLeptons_isElectron': 'l1(isElectron)', 'selectedJets_nominal_pt': 'pT(leading jet)',
			'selectedJets_nominal_eta': '|eta(leading jet)|', 'subselectedJets_nominal_pt': 'pT(subleading jet)',
			'subselectedJets_nominal_eta': '|eta(subleading jet)|'
			}

df.rename(rename_dict, axis=1, inplace=True)

original_list = model.get_booster().feature_names
renamed_list = [rename_dict[name] for name in original_list]

# plot feature importance
importances = {key:model.feature_importances_[i] for i, key in enumerate(df.keys())}
importances = {key: val for key, val in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
print(importances)
fig, ax = pyplot.subplots()
pyplot.bar(range(len(importances.keys())), [*importances.values()])
pyplot.xticks(range(len(importances.keys())), importances.keys(), rotation=90)
pyplot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.5)
pyplot.title('XGBoost feature importances - '+str(args.year))
pyplot.ylabel('Feature importances score')
pyplot.savefig(os.path.join(path2,'BDT_feature_importances_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path2,'BDT_feature_importances_'+str(args.year)+'.png'))
pyplot.clf()

model.get_booster().feature_names = renamed_list

importance_gain = model.get_booster().get_score(importance_type="gain")
importance_weight = model.get_booster().get_score(importance_type="weight")
importance_cover= model.get_booster().get_score(importance_type="cover")

for key in importance_gain.keys():
	importance_gain[key] = round(importance_gain[key],2)
	importance_cover[key] = round(importance_cover[key],2)


#fig, ax = pyplot.subplots(5,8)
ax = plot_importance(importance_gain, importance_type='gain')
pyplot.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
pyplot.title('XGBoost feature importances (gain) - '+str(args.year))
#pyplot.show()
pyplot.tight_layout()
pyplot.savefig(os.path.join(path2,'BDT_feature_importances_fscore_gain_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path2,'BDT_feature_importances_fscore_gain_'+str(args.year)+'.png'))
pyplot.clf()

#fig, ax = pyplot.subplots(5,8)
ax = plot_importance(importance_weight, importance_type='weight')
pyplot.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
pyplot.title('XGBoost feature importances (weight) - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(path2,'BDT_feature_importances_fscore_weight_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path2,'BDT_feature_importances_fscore_weight_'+str(args.year)+'.png'))
pyplot.clf()

#fig, ax = pyplot.subplots(5,8)
ax = plot_importance(importance_cover, importance_type='cover')
pyplot.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
pyplot.title('XGBoost feature importances (cover) - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(path2,'BDT_feature_importances_fscore_cover_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(path2,'BDT_feature_importances_fscore_cover_'+str(args.year)+'.png'))


