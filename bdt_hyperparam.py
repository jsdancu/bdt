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

today = date.today()

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--oneFile', dest='oneFile', action='store',type=int, default=1)
#parser.add_argument('--dir_data', dest='dir_data', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/bdt/bdt_df_2016.pkl")
parser.add_argument('--dir_src', dest='dir_src', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--bdt_inputs', dest='bdt_inputs', action='store',default="bdt/bdt_inputs_hyperparam.json")
parser.add_argument('--combination', dest='combination', action='store',type=int, default=0)

args = parser.parse_args()

path = args.dir_src

if args.oneFile:
	data = path+"bdt/bdt_df_"+str(args.year)+"_smallie.pkl"
else:
	data = path+"bdt/bdt_df_"+str(args.year)+".pkl"
bdt_inputs_file = path+args.bdt_inputs

bdt_training_dir = path+"bdt/bdt_training_hyperparam_"+str(today.strftime("%Y%m%d"))+"_"+str(args.year)+"_"+str(args.combination)

if os.path.exists(bdt_training_dir):
	print("Overwriting dir")
else:
	os.mkdir(bdt_training_dir)

df_biggie = pd.read_pickle(data)
print(df_biggie)

array_bdt = json.load(open(bdt_inputs_file))[str(args.combination)]

print("array_bdt: ", array_bdt)

df = df_biggie[array_bdt+["genweight"]]
print(df)

label = df_biggie["label"]

#split data into test and train, including genweight
X_train1, X_test1, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=42, stratify=label)

#separate test and train data from genweight column
X_train = X_train1[array_bdt]
X_test = X_test1[array_bdt]
genweight = X_train1["genweight"]
df = df[array_bdt]
df = df.reindex(sorted(df.columns), axis=1)
print(df)

# fit model to training data
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
model._Booster.save_model(os.path.join(bdt_training_dir,'bdt_'+str(args.year)+'.model'))

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.xlabel('Epochs')
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_loss_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_loss_'+str(args.year)+'.png'))

# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.xlabel('Epochs')
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_error_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_error_'+str(args.year)+'.png'))

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
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_output_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_output_'+str(args.year)+'.png'))

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

categories={"muonmuon": r'$\mu \mu$', "muonelectron": r'$\mu$ e', 
            "electronelectron": r'e $\mu$', "electronmuon": 'ee'}
for category in categories.keys():
    cut = category_cut(X_test, category)
    fig, axes = pyplot.subplots()
    pyplot.hist(probs[cut][y_test[cut]==0], label='background', bins = 50, histtype='step')
    pyplot.hist(probs[cut][y_test[cut]==1], label='signal', bins = 50, histtype='step')
    axes.legend()
    pyplot.ylabel('Events')
    pyplot.xlabel('signal (1) vs background (0)')
    pyplot.title('BDT separation - '+str(args.year))
    pyplot.savefig(os.path.join(bdt_training_dir,'BDT_output_'+category+"_"+str(args.year)+'.pdf'))
    pyplot.savefig(os.path.join(bdt_training_dir,'BDT_output_'+category+"_"+str(args.year)+'.png'))


# fpr means false-positive-rate
# tpr means true-positive-rate
fpr, tpr, _ = metrics.roc_curve(y_test, probs)

#auc_score = 1.0 - metrics.auc(fpr, tpr)
auc_score = metrics.auc(tpr, fpr)
print('AUC = {:.3f}'.format(auc_score))#want it as small as possible

#10% background efficiency line
fpr_interpolated = scipy.interpolate.interp1d(fpr, tpr)
fpr_mark = float(fpr_interpolated(0.1))
print(str('TPR at which FPR=10% : {:.3f}'.format(fpr_mark))+"\n")

fig, ax = pyplot.subplots()
ax.plot(tpr, fpr, label='AUC = {:.3f}'.format(auc_score))
ax.legend(loc='upper left')
pyplot.yscale('log')
pyplot.hlines(1e-1, 0, fpr_mark, linestyle="dashed")
pyplot.vlines(fpr_mark, 1e-4, 1e-1, linestyle="dashed")
pyplot.xlabel('Signal Efficiency')
pyplot.ylabel('Background Efficiency')
pyplot.title('XGBoost ROC curve - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_roc_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_roc_'+str(args.year)+'.png'))
pyplot.clf()

fig, ax = pyplot.subplots()
for category in categories.keys():
    cut = category_cut(X_test, category)
    fpr_cat, tpr_cat, _ = metrics.roc_curve(y_test[cut], probs[cut])
    ax.plot(tpr_cat, fpr_cat, label=categories[category])

    fpr_interpolated = scipy.interpolate.interp1d(fpr_cat, tpr_cat)
    fpr_mark = float(fpr_interpolated(0.1))
    pyplot.vlines(fpr_mark, 1e-4, 1e-1, linestyle="dashed")

ax.legend(title='Dilepton categories', loc='upper left')
pyplot.yscale('log')
pyplot.hlines(1e-1, 0, fpr_mark, linestyle="dashed")
pyplot.xlabel('Signal Efficiency')
pyplot.ylabel('Background Efficiency')
pyplot.title('BDT ROC curve - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_roc_categories_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_roc_categories_'+str(args.year)+'.png'))
pyplot.clf()

pickle.dump(fig, open(os.path.join(bdt_training_dir,'BDT_roc_'+str(args.year)+'.fig.pickle'), 'wb'))

rename_dict = {
			"EventObservables_nominal_met": "$p^{miss}_{T}$", 
            "EventObservables_nominal_ht": "$H_{T}$",
            "dilepton_mass": "$m(l_{1}l_{2})$", 
            "dilepton_deltaPhi": "$|\\Delta\\phi(l_{1}l_{2})|$", 
            "dilepton_deltaR": "$\\Delta R(l_{1}l_{2})$", 
            "dilepton_charge": "dilepton charge",
            "leadingLeptons_nominal_mtw": "$m_{T}$", 
            "leadingLeptons_nominal_deltaPhi": "$|\\Delta\\phi(l_{1},p^{miss}_{T})|$",
            "leadingLeptons_pt": "$p_{T}(l_{1})$", 
            "leadingLeptons_eta": "$|\\eta(l_{1})|$", 
            "subleadingLeptons_pt": "$p_{T}(l_{2})$", 
            "subleadingLeptons_eta": "$|\\eta(l_{2})|$", 
            "selectedJets_nominal_pt": "$p_{T}$(leading jet)", 
            "selectedJets_nominal_eta": "|$\\eta$(leading jet)|", 
            "subselectedJets_nominal_pt": "$p_{T}$(subleading jet)", 
            "subselectedJets_nominal_eta": "|$\\eta$(subleading jet)|", 
            "leadingLeptons_isElectron": "isElectron ($l_{1}$)", 
            "subleadingLeptons_isElectron": "isElectron ($l_{2}$)"
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
pyplot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
pyplot.title('XGBoost feature importances - '+str(args.year))
pyplot.ylabel('Feature importances score')
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_feature_importances_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_feature_importances_'+str(args.year)+'.png'))
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
pyplot.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
pyplot.title('XGBoost feature importances (gain) - '+str(args.year))
#pyplot.show()
pyplot.tight_layout()
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_feature_importances_fscore_gain_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_feature_importances_fscore_gain_'+str(args.year)+'.png'))
pyplot.clf()

#fig, ax = pyplot.subplots(5,8)
ax = plot_importance(importance_weight, importance_type='weight')
pyplot.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
pyplot.title('XGBoost feature importances (weight) - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_feature_importances_fscore_weight_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_feature_importances_fscore_weight_'+str(args.year)+'.png'))
pyplot.clf()

#fig, ax = pyplot.subplots(5,8)
ax = plot_importance(importance_cover, importance_type='cover')
pyplot.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
pyplot.title('XGBoost feature importances (cover) - '+str(args.year))
#pyplot.show()
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_feature_importances_fscore_cover_'+str(args.year)+'.pdf'))
pyplot.savefig(os.path.join(bdt_training_dir,'BDT_feature_importances_fscore_cover_'+str(args.year)+'.png'))

#Book keeping of BDT performance
text_file_object = open(os.path.join(bdt_training_dir,'BDT_model_performance.txt'), 'w')
text_file_object.write(str("Accuracy = %.2f%%" % (accuracy * 100.0))+"\n")
text_file_object.write(str('log loss = {:.3f}'.format(log_loss))+"\n")
text_file_object.write(str('AUC = {:.3f}'.format(auc_score))+"\n")
text_file_object.write(str('TPR at which FPR=10% : {:.3f}'.format(fpr_mark))+"\n")
text_file_object.close()

bdt_performance = {}
bdt_performance["Accuracy"] = accuracy * 100.0 
bdt_performance['log loss'] = log_loss
bdt_performance['AUC'] = auc_score
bdt_performance['TPR at which FPR=10%'] = fpr_mark
with open(os.path.join(bdt_training_dir,'BDT_model_performance.json'), 'w') as json_file_object:
    json.dump(bdt_performance, json_file_object, indent=1)


