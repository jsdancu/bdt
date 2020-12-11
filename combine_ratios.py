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
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--var', dest='var', action='store',default="EventObservables_nominal_met")
parser.add_argument('--dir', dest='dir', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--luminosity', dest='luminosity', action='store',type=float, default=35.88)

args = parser.parse_args()

CR = "deltaR_SR"
CR_var_latex = r"\min(\Delta R(l_{2}, \mathrm{jet}))<1.3"
categories = ["muonmuon", "muonelectron", "electronelectron", "electronmuon"]
dilepton_charge = ["SS+OS", "SS", "OS"]
'''
mcs = {
        "T*GJets*": ["#40E0D0", r'$t(\bar{t})\gamma$'], 
        "WZTo*": ["#b638d6", r'WZ'], 
        "ST*": ["#ffb3bf", 'single-top'], 
        "TTTo*": ["#ef5350", r'$t\bar{t}$'], 
        "[WZ]GTo*": ["#00FA9A", r'$V\gamma$'], 
        "QCD*": ["#bdbdbd", 'QCD'], 
        "W*J_*": ["#388e3c", r'$W \rightarrow l\nu$'], 
        "DY*": ["#1976d2", r'DY$\rightarrow ll$']
        }
'''
mcs = ["T*GJets*", "WZTo*", "ST*", "TTTo*", "[WZ]GTo*", "QCD*", "W*J_*", "DY*"]

colours = ["#40E0D0", "#b638d6", "#ffb3bf", "#ef5350", "#00FA9A", "#bdbdbd", "#388e3c", "#1976d2"]


file_names = []
for category in categories:
        for dilep_ch in dilepton_charge:
            file_names.append(args.dir+"plots/"+args.var+"_"+CR+"_"+category+"_"+dilep_ch+"_"+args.year+".txt")

ratios = {dilep_ch: {mc: 0.0 for mc in mcs} for dilep_ch in dilepton_charge}

for file_name in file_names:
    for dilep_ch in dilepton_charge:
        if "_"+dilep_ch+"_" in file_name:
            with open(file_name) as f: 
                for i, l in enumerate(f):
                    if i > 1:
                        ratios[dilep_ch][eval(l)[0]] += eval(l)[1] 
            
print(ratios) 
            
def pie_chart(integrals, dilep_ch, type):

    mcs_legend = [r'$t(\bar{t})\gamma$', r'WZ', 'single-top', r'$t\bar{t}$', r'$V\gamma$', 'QCD', r'$W \rightarrow l\nu$', r'DY$\rightarrow ll$']

    #plt.style.use([hep.style.ROOT, hep.style.firamath])
    fig, ax = pyplot.subplots()
    
    lumi = ax.text(0.7, 1.2, r"%.2f fb$^{-1}$ (%s)" % (args.luminosity, args.year), fontsize=14)
    cms = ax.text(-1.6, 1.2, u"CMS $\it{Simulation Preliminary}$",fontsize=16, fontweight='bold')
    cat = ax.text(-1.6, -1.25, r"%s, %s, $%s$" % ("all categories", dilep_ch, CR_var_latex), fontsize=14)
    
    # A standard pie plot
    patches, texts, autotexts = ax.pie(integrals, labels=mcs_legend, autopct='%1.1f%%', colors=colours, shadow=True)
    pyplot.axis('equal')

    pyplot.savefig(os.path.join(args.dir,"plots/"+args.var+"_"+str(type)+"_"+str(CR)+"_allcategory_"+str(dilep_ch)+"_"+str(args.year)+".pdf"))
    pyplot.savefig(os.path.join(args.dir,"plots/"+args.var+"_"+str(type)+"_"+str(CR)+"_allcategory_"+str(dilep_ch)+"_"+str(args.year)+".png"))

columns = "MC background type --- Integral --- Event weight"

for dilep_ch in dilepton_charge:

    text_file_object = open(os.path.join(args.dir+"plots/"+args.var+"_"+str(CR)+"_allcategory_"+str(dilep_ch)+"_"+str(args.year)+'.txt'), 'w')
    process = [args.var, CR, "allcategory", dilep_ch, args.year]
    text_file_object.write(str(process)+"\n")
    text_file_object.write(columns+"\n")

    integrals = []
    for mc in mcs:
        integrals.append(ratios[dilep_ch][mc])
        output = [mc, ratios[dilep_ch][mc], "N/A"]
        text_file_object.write(str(output)+"\n")

    pie_chart(integrals, dilep_ch, "piechart_reweighted")
    text_file_object.close()