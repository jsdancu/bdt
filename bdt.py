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
		#print(path)

		data = data[data.nleadingLepton == 1]#one leading lepton in event
		data = data[data.nsubleadingLepton == 1]#one subleading lepton in event
		data = data[data.nselectedJets_nominal > 0]#at least one jet in event
		data = data[data.dilepton_mass < 80.]#apply CR cut
		data = data[data.dilepton_mass > 20.]#apply CR cut
		data = data[data.EventObservables_nominal_met < 100.]#apply CR cut
		#data = data[data.lepJet_nominal_deltaR < 0.4]#DeltaR cut
		data = data[data.dilepton_charge == -1]#Dirac samples only, i.e. opposite sign leptons
		#data = data[data.IsoMuTrigger_flag == True]#muon trigger applied for event

		iters.append(data)
		# add cuts

	return pd.concat(iters)

path = "/vols/cms/jd918/LLP/CMSSW_10_2_18/"

array_preselection = [
			"nleadingLepton", "nlepJet_nominal", "nsubleadingLepton", "lepJet_nominal_deltaR",
			"dilepton_charge", "IsoMuTrigger_flag"
			]

with open(os.path.join(path,"src/PhysicsTools/NanoAODTools/data/bdt/bdt_inputs.txt")) as f:
	array_bdt = [line.rstrip() for line in f]

array_list = array_preselection + array_bdt

n_events=300000

sig_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016//HNL_*/nano_*_Friend.root"), array_list, True)

wjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/WToLNu_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
tt_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/TTToSemiLeptonic_*/nano_*_Friend.root"), array_list, False).sample(n=n_events)
dyjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/DYJetsToLL*amcatnlo*/nano_*_Friend.root"), array_list, False).sample(n=n_events)

'''
sig_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/HNL_dirac_all_ctau1p0e01_massHNL4p5_Vall4p549e-03-2016/nano_1_Friend.root"), array_list, True)
wjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/WToLNu_0J_13TeV-amcatnloFXFX-pythia8-2016/nano_1_Friend.root"), array_list, False)
tt_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/TTToSemiLeptonic_*/nano_1_Friend.root"), array_list, False)
dyjets_df = get_df(os.path.join(path,"src/nanoAOD_friends_200622/2016/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-2016/nano_1_Friend.root"), array_list, False)
#bkg_df = pd.concat([wjets_df, dyjets_df, tt_df]).sample(n=n_events)
'''
print(sig_df.shape[0])

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
		print(df[feat])
		df[feat] = df[feat].map(lambda x: x[0])
		print(df[feat])
'''
df["EventObservables_nominal_met_phi-leadingLepton_phi"] = df["EventObservables_nominal_met_phi"] - df["leadingLepton_phi"]
df.drop("leadingLepton_phi", axis=1, inplace=True)
print(df)
'''

df = df.reindex(sorted(df.columns), axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=42, stratify=label)

# fit model no training data
model = XGBClassifier(objective='binary:logistic', learning_rate=0.1, max_depth=10, n_estimators=200, nthread=-1)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
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

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
#pyplot.show()
pyplot.savefig('BDT_loss.pdf')
pyplot.savefig('BDT_loss.png')

# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
#pyplot.show()
pyplot.savefig('BDT_error.pdf')
pyplot.savefig('BDT_error.png')

# take the second column because the classifier outputs scores for
# the 0 class as well
probs = model.predict_proba(X_test)[:, 1]

fig, axes = pyplot.subplots()
pyplot.hist(probs[y_test==0], label='background')
pyplot.hist(probs[y_test==1], label='signal')
axes.legend()
pyplot.ylabel('BDT output')
pyplot.xlabel('signal (1) vs background (0)')
pyplot.title('XGBoost separation')
pyplot.savefig('BDT_output.pdf')
pyplot.savefig('BDT_output.png')

# fpr means false-positive-rate
# tpr means true-positive-rate
fpr, tpr, _ = metrics.roc_curve(y_test, probs)

auc_score = metrics.auc(fpr, tpr)
print('AUC = {:.3f}'.format(auc_score))

fig, ax = pyplot.subplots()
ax.plot(tpr, fpr, label='AUC = {:.3f}'.format(auc_score))
ax.legend(loc='lower right')
pyplot.yscale('log')
pyplot.xlabel('Signal Efficiency')
pyplot.ylabel('Background Efficiency')
pyplot.title('XGBoost ROC curve')
#pyplot.show()
pyplot.savefig('BDT_roc.pdf')
pyplot.savefig('BDT_roc.png')

pickle.dump(fig, open('BDT_roc.fig.pickle', 'wb'))

# plot feature importance
importances = {key:model.feature_importances_[i] for i, key in enumerate(df.keys())}
importances = {key: val for key, val in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
print(importances)
fig, ax = pyplot.subplots()
pyplot.bar(range(len(importances.keys())), [*importances.values()])
pyplot.xticks(range(len(importances.keys())), importances.keys(), rotation=90)
pyplot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.5)
pyplot.title('XGBoost feature importances')
pyplot.ylabel('Feature importances score')
pyplot.savefig('BDT_feature_importances.pdf')
pyplot.savefig('BDT_feature_importances.png')

fig, ax = pyplot.subplots(5,8)
plot_importance(model)
pyplot.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1)
pyplot.title('XGBoost feature importances')
#pyplot.show()
pyplot.savefig('BDT_feature_importances_fscore.pdf')
pyplot.savefig('BDT_feature_importances_fscore.png')

model._Booster.save_model(os.path.join(path,"src/PhysicsTools/NanoAODTools/data/bdt/bdt.model"))

'''
#iris = sns.load_dataset(\"iris\")
g = sns.pairplot(df, vars = df.keys())
g.savefig('BDT_feature_correlations.pdf')
g.savefig('BDT_feature_correlations.png')
'''
#model._Booster.save_model(os.path.join(path,"src/bdt.model"))
