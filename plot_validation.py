import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as col

import numpy as np
import math
import ROOT
ROOT.gROOT.SetBatch(1)
import pandas as pd
import uproot
import random
from array import array

import subprocess
import re
import time
import json
import yaml
import pickle
import ast
import glob
import re
import fnmatch

import warnings

warnings.simplefilter(action = "ignore", category = RuntimeWarning)

import style
style.makeColorTable()

#import mplhep as hep

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--CR', dest='CR', action='store',default="deltaR_CR")
parser.add_argument('--category', dest='category', action='store',default="muonmuon")
parser.add_argument('--dilepton_charge', dest='dilepton_charge', action='store',default="SS+OS")
parser.add_argument('--var', dest='var', action='store',default="bdt_score_nominal")
parser.add_argument('--xaxis_title', dest='xaxis_title', action='store',default="BDT score")
parser.add_argument('--dir', dest='dir', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--dir_ntuples', dest='dir_ntuples', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/nanoAOD_friends_200622")
parser.add_argument('--dir_LLP', dest='dir_LLP', action='store',default="/vols/cms/LLP/")
parser.add_argument('--sm_xsec', dest='sm_xsec', action='store',default="xsec.json")
parser.add_argument('--hnl_xsec', dest='hnl_xsec', action='store',default="gridpackLookupTable.json")
parser.add_argument('--dir_yields', dest='dir_yields', action='store',default="yields_201117")
parser.add_argument('--file_yields', dest='file_yields', action='store',default="eventyields.json")
parser.add_argument('--file_hnl_yields', dest='file_hnl_yields', action='store',default="eventyieldsHNL.json")
parser.add_argument('--luminosity', dest='luminosity', action='store',type=float, default=35.92)
parser.add_argument('--feature_min', dest='feature_min', action='store',type=float, default=0.0)
parser.add_argument('--feature_max', dest='feature_max', action='store',type=float, default=1.0)
parser.add_argument('--bins', dest='bins', action='store',type=int, default=50)
parser.add_argument('--array_var', dest='array_var', action='store',type=int, default=0)
parser.add_argument('--log_scale', dest='log_scale', action='store',type=int, default=1)
parser.add_argument('--oneFile', dest='oneFile', action='store',type=int, default=1)
parser.add_argument('--hnl', dest='hnl', action='store',default="HNL_dirac_all_ctau1p0e01_massHNL10p0_Vall5p262e-04-*")

args = parser.parse_args()

coupling_dict = {}
coupling_dict["muonmuon"] = 12
coupling_dict["muonelectron"] = 7
coupling_dict["electronmuon"] = 7
coupling_dict["electronelectron"] = 2
category = args.category
coupling = str(coupling_dict[category])

category_name_dict = {}
category_name_dict["muonmuon"] = "#mu#mu"
category_name_dict["muonelectron"] = "#mue"
category_name_dict["electronmuon"] = "e#mu"
category_name_dict["electronelectron"] = "ee"
category_name = category_name_dict[category]

category_name_latex_dict = {}
category_name_latex_dict["muonmuon"] = "\\mu\\mu"
category_name_latex_dict["muonelectron"] = "\\mu e"
category_name_latex_dict["electronmuon"] = "e\\mu"
category_name_latex_dict["electronelectron"] = "ee"
category_name_latex = category_name_latex_dict[category]


CR_var = ""
if args.CR == "deltaR_CR":
    CR_var = "min(#DeltaR(l_{2}, jet))>1.3"
    CR_var_latex = r"\min(\Delta R(l_{2}, \mathrm{jet}))>1.3"
elif args.CR == "deltaR_SR":
    CR_var = "min(#DeltaR(l_{2}, jet))<1.3"
    CR_var_latex = r"\min(\Delta R(l_{2}, \mathrm{jet}))<1.3"

print(args.dir_ntuples)
def read_data(type):

    names = ROOT.std.vector('string')()

    for proc in filelist:
        if type not in proc:
            continue
        with open(os.path.join(files, proc)) as f:
            content = f.readlines()
        for n in content: names.push_back(n.strip())

    return names

def define_plots_1D(binning1, xaxis_title, yaxis_title, title):

    hist1 = ROOT.TH1D(title, title, *binning1)

    hist1.GetXaxis().SetTitle(xaxis_title)
    hist1.GetYaxis().SetTitle(yaxis_title)
    hist1.GetXaxis().SetTitleOffset(0.8)
    hist1.GetYaxis().SetTitleOffset(0.8)

    return hist1

def plotting_1D(array1, hist1, puWeight):

    ntimes = len(array1)

    hist1.FillN(ntimes, array('d', array1), array('d', puWeight))

    return hist1

def selection_highMET(tree):

    sel = (tree.array("nleadingLepton") == 1) & \
          (tree.array("nsubleadingLepton") == 1) & \
          (tree.array("EventObservables_nominal_met") > 100.) & \
          (tree.array("dilepton_mass") < 80.) & \
          (tree.array("dilepton_mass") > 20.) & \
          (tree.array("MET_filter") == 1)

    return sel

def selection_DY(tree):

    sel = (tree.array("nleadingLepton") == 1) & \
          (tree.array("nsubleadingLepton") == 1) & \
          (tree.array("dilepton_mass") > 90.) & \
          (tree.array("dilepton_mass") < 110.) & \
          (tree.array("MET_filter") == 1)

    return sel

def selection_deltaR_CR(tree):

    sel = (tree.array("nleadingLeptons") == 1) & \
          (tree.array("nsubleadingLeptons") == 1) & \
          (tree.array("nselectedJets_nominal") > 0) & \
          (tree.array("nselectedJets_nominal") < 5) & \
          (tree.array("category_simplified_nominal_index") < 0) & \
          (tree.array("MET_filter") == 1) & \
          (tree.array("dilepton_mass") < 91.1876-15.) & \
          (tree.array("dilepton_mass") > 20.) & \
          (tree.array("EventObservables_nominal_met") < 100.)
          #(map(lambda x: (x < 1.3).any(), tree.array("selectedJets_nominal_minDeltaRSubtraction"))) & \

    if args.dilepton_charge == "SS":
        sel = sel & (tree.array("dilepton_charge") == 1)
    elif args.dilepton_charge == "OS":
        sel = sel & (tree.array("dilepton_charge") == -1)

    return sel

def selection_deltaR_SR(tree):

    sel = (tree.array("nleadingLeptons") == 1) & \
          (tree.array("nsubleadingLeptons") == 1) & \
          (tree.array("nselectedJets_nominal") > 0) & \
          (tree.array("nselectedJets_nominal") < 5) & \
          (tree.array("category_simplified_nominal_index") > 0) & \
          (tree.array("MET_filter") == 1) & \
          (tree.array("dilepton_mass") < 91.1876-15.) & \
          (tree.array("dilepton_mass") > 20.) & \
          (tree.array("EventObservables_nominal_met") < 100.)
          #(map(lambda x: (x < 1.3).any(), tree.array("selectedJets_nominal_minDeltaRSubtraction"))) & \

    if args.dilepton_charge == "SS":
        sel = sel & (tree.array("dilepton_charge") == 1)
    elif args.dilepton_charge == "OS":
        sel = sel & (tree.array("dilepton_charge") == -1)

    return sel

def selection_preselection(tree):
    sel = (tree.array("nleadingLeptons") == 1) & \
          (tree.array("nsubleadingLeptons") == 1) & \
          (tree.array("nselectedJets_nominal") > 0) & \
          (tree.array("nselectedJets_nominal") < 5) & \
          (tree.array("MET_filter") == 1) & \
          (tree.array("dilepton_mass") < 85.) & \
          (tree.array("dilepton_mass") > 20.) & \
          (tree.array("EventObservables_nominal_met") < 100.)
          #(tree.array("lepJet_nominal_deltaR") < 2.0) & \
          #(tree.array("dilepton_charge") == -1)#Dirac samples OS
    if args.dilepton_charge == "SS":
        sel = sel & (tree.array("dilepton_charge") == 1)
    elif args.dilepton_charge == "OS":
        sel = sel & (tree.array("dilepton_charge") == -1)

    return sel

def dilepton_categories(category, tree):
    if category == "muonmuon":
        sel = (tree.array("Leptons_muonmuon")==1) & \
              (tree.array("IsoMuTrigger_flag") == 1)
    elif category == "muonelectron":
        sel = (tree.array("Leptons_muonelectron")==1) & \
              (tree.array("IsoMuTrigger_flag") == 1)
    elif category == "electronelectron":
        sel = (tree.array("Leptons_electronelectron")==1) & \
              (tree.array("IsoElectronTrigger_flag") == 1)
    elif category == "electronmuon":
        sel = (tree.array("Leptons_electronmuon")==1) & \
              (tree.array("IsoElectronTrigger_flag") == 1)

    return sel

directory_ntuples = os.path.join(args.dir_ntuples, args.year)
dirlist_ntuples = os.listdir(directory_ntuples)

xsecs_raw = json.load(open(os.path.join(args.dir_LLP, args.sm_xsec)))
xsecs_hnl_raw = json.load(open(os.path.join(args.dir_LLP, args.hnl_xsec)))
#print xsecs_raw

yields_raw = json.load(open(os.path.join(args.dir_LLP, args.dir_yields, args.year, args.file_yields)))
yields_hnl_raw = json.load(open(os.path.join(args.dir_LLP, args.dir_yields, args.year, args.file_hnl_yields)))
#print yields_raw

#mcs = ["T*GJets*", "WZTo*", "ST*", "TTTo*", "[WZ]GTo*", "QCD*", "DY*", "W*J_*", args.hnl]
mcs = ["TTTo*", "[WZ]GTo*", "QCD*", "DY*", "W*J_*", args.hnl]

#colours = ["#40E0D0", "#b638d6", "#ffb3bf", "#ef5350", "#00FA9A", "#bdbdbd", "#1976d2", "#388e3c"]
colours = ["#ef5350", "#ffb3bf", "#bdbdbd", "#1976d2", "#388e3c"]

files_MC = {}
files_data = {}
xsecs = {}
yields = {}

processes = []
for mc in mcs:
    directories = glob.glob(directory_ntuples+"/"+mc)
    print mc, directories
    processes.extend(directories)
print "processes: ", processes

for dir in dirlist_ntuples:
    #dir = dir.split("/")[-1]
    if any([dir in proc for proc in processes]):
    #if any([dir.startswith(mc) for mc in mcs]):
        if "DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-2016" in dir:
            continue
        else:
            files_MC[dir] = os.listdir(os.path.join(directory_ntuples, dir))
            #files_MC[dir] = os.listdir(dir)
            if "HNL" in dir:
                xsecs[dir] = 1000. * xsecs_hnl_raw[dir.replace("-"+str(args.year), "")]["weights"][coupling]["xsec"]["nominal"]
            else:
                for xsec_key in xsecs_raw.keys():
                    if xsec_key in dir:
                        xsecs[dir] = xsecs_raw[xsec_key]

            if "HNL" in dir:
                yields[dir] = yields_hnl_raw[dir]["LHEWeights_coupling_"+str(coupling)]
            else:
                for yield_key in yields_raw.keys():
                    if yield_key in dir:
                        yields[dir] = yields_raw[yield_key]["weighted"]
                        break


    if category[0] == "m":
        if any(["SingleMuon" in dir]):
            files_data[dir] = os.listdir(os.path.join(directory_ntuples, dir))
    elif category[0] == "e":
        if any(["SingleElectron" in dir]) or any(["EGamma" in dir]):
            files_data[dir] = os.listdir(os.path.join(directory_ntuples, dir))

binning = [args.bins, args.feature_min, args.feature_max]
#print binning
#print(type(binning))

hists_reweighted = {key:[define_plots_1D(binning, args.var, "Entries", "samples reweighted"+key+str(i)) for i in range(3)] for key in files_MC.keys()}
hist_reweighted = {key: [define_plots_1D(binning, args.var, "Entries", "MC reweighted"+key+str(i)) for i in range(3)] for key in mcs}
hists_original = {key:[define_plots_1D(binning, args.var, "Entries", "original"+key+str(random.randint(1, 1000000)))] for key in files_MC.keys()}
hist_original = {key:[define_plots_1D(binning, args.var, "Entries", "original"+key+str(random.randint(1, 1000000)))] for key in mcs}
hists_data = {key:[define_plots_1D(binning, args.var, "Entries", "data"+key+str(random.randint(1, 1000000)))] for key in files_data.keys()}
hist_data = define_plots_1D(binning, args.var, "Entries", "data"+str(random.randint(1, 1000000)))

for dir, files in files_MC.iteritems():

    print dir

    weight = eval(str(xsecs[dir])) * args.luminosity * 1e3 / yields[dir]

    if args.oneFile:
        files1 = [files[1]]
    else:
        files1 = files

    for file in files1:
        #print file
        try:
            f_MC = uproot.open(os.path.join(directory_ntuples, dir, file))
        except ValueError:
            print("Could not open file.")
            continue

        if not f_MC:
            print "could not open file: ", dir, file
            continue

        #print f_MC.keys()

        tree = f_MC["Friends"]
    	if not tree:
    	    print "empty tree in: ", dir, file
    	    continue

        if args.CR == "DY":
            cuts = selection_DY(tree)
        elif args.CR == "highMET":
            cuts = selection_highMET(tree)
        elif args.CR == "deltaR_CR":
            cuts = selection_deltaR_CR(tree)
        elif args.CR == "deltaR_SR":
            cuts = selection_deltaR_SR(tree)
        elif args.CR == "preselection":
            cuts = selection_preselection(tree)

        category_weight = dilepton_categories(args.category, tree)
        cuts = cuts*category_weight

        if args.array_var == 1:
            tree_cuts = np.array([i[0] if len(i) > 0 else None for i in tree.array(args.var)])[cuts != 0]
            tree_cuts = tree_cuts[tree_cuts != np.array(None)]
        else:
            tree_cuts = tree.array(args.var)[cuts != 0].flatten()

        genweights = tree.array("genweight")
        id_iso_weights = tree.array("tightMuon_weight_iso_nominal")*tree.array("tightMuon_weight_id_nominal")*tree.array("tightElectron_weight_id_nominal")*tree.array("tightElectron_weight_reco_nominal")*tree.array("looseElectrons_weight_reco_nominal")

        if category[0] == "m":
            trigger_weight = tree.array("IsoMuTrigger_weight_trigger_nominal")*tree.array("IsoMuTrigger_flag")
        elif category[0] == "e":
            trigger_weight = tree.array("IsoElectronTrigger_flag")

        all_weights = (genweights*id_iso_weights*trigger_weight)[cuts != 0].flatten()

        if len(tree_cuts) > 0:
            for i, var in enumerate(["_up", "_nominal", "_down"]):
                puweight_cuts = tree.array("puweight"+var)[cuts != 0].flatten()
                hist_var_reweighted = define_plots_1D(binning, args.var, "Entries", args.var+" reweighted"+str(random.randint(1, 10000)))
                hist_var_reweighted = plotting_1D(tree_cuts, hist_var_reweighted, weight * np.multiply(all_weights, puweight_cuts))
                hists_reweighted[dir][i] += hist_var_reweighted

            hist_var_original = define_plots_1D(binning, args.var, "Entries", args.var+str(random.randint(1, 1000000)))
            hist_var_original = plotting_1D(tree_cuts, hist_var_original, weight * all_weights)
            hists_original[dir][0] += hist_var_original

for mc in mcs:
    print "mc: ", mc
    for dir in files_MC.keys():
        #print "mc: ", mc
        #print "dir: ", dir
        if bool(re.search(fnmatch.translate(mc), dir)):
            print "dir: ", dir
            print "re.search(mc, dir): ", re.search(mc, dir)
            for i, var in enumerate(["_up", "_nominal", "_down"]):
                hist_reweighted[mc][i].Add(hists_reweighted[dir][i])
                #map(lambda hist_reweighted[mc]: hist_reweighted[mc] + hists_reweighted['dir'], hists_reweighted)
            hist_original[mc][0].Add(hists_original[dir][0])
        

if args.CR == "deltaR_CR":
    for dir, files in files_data.iteritems():

        print dir

        if args.oneFile:
            files1 = [files[1]]
        else:
            files1 = files

        for file in files1:
            #print file
            f_data = uproot.open(os.path.join(directory_ntuples, dir, file))
            if not f_data:
                print "could not open file: ", dir, file
                continue

            #print f_data.keys()
            tree = f_data["Friends"]
            if not tree:
        	    print "empty tree in: ", dir, file
        	    continue

            if args.CR == "DY":
                cuts = selection_DY(tree)
            elif args.CR == "highMET":
                cuts = selection_highMET(tree)
            elif args.CR == "deltaR_CR":
                cuts = selection_deltaR_CR(tree)
            elif args.CR == "deltaR_SR":
                cuts = selection_deltaR_SR(tree)
            elif args.CR == "preselection":
                cuts = selection_preselection(tree)

            category_weight = dilepton_categories(args.category, tree)
            cuts = cuts*category_weight

            if args.array_var == 1:
                tree_cuts = np.array([i[0] if len(i) > 0 else None for i in tree.array(args.var)])[cuts != 0]
                tree_cuts = tree_cuts[tree_cuts != np.array(None)]
            else:
                tree_cuts = tree.array(args.var)[cuts != 0].flatten()

            if category[0] == "m":
                trigger_weight = tree.array("IsoMuTrigger_flag")[cuts != 0].flatten()
            elif category[0] == "e":
                trigger_weight = tree.array("IsoElectronTrigger_flag")[cuts != 0].flatten()

            if len(tree_cuts) > 0:

                hist_var = define_plots_1D(binning, args.var, "Entries", args.var+str(random.randint(1, 1000000)))
                hist_var = plotting_1D(tree_cuts, hist_var, trigger_weight)
                hists_data[dir][0].Add(hist_var)

        hist_data.Add(hists_data[dir][0])

hist_stack_reweighted = ROOT.THStack("MC reweighted", "MC reweighted")
hist_stack_original = ROOT.THStack("MC original", "MC original")

process = [args.var, args.CR, category, args.dilepton_charge, args.year]
columns = "MC background type --- Integral --- Event weight"

print process
print columns

pickle_file_object = open(os.path.join(args.dir+"plots/"+args.var+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+'.pkl'), 'wb')
text_file_object = open(os.path.join(args.dir+"plots/"+args.var+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+'.txt'), 'w')

pickle.dump(process, pickle_file_object)
text_file_object.write(str(process)+"\n")
text_file_object.write(columns+"\n")

hist_integrals = []

for mc, col in zip(mcs, colours):
    if "HNL" in mc:
        continue
    hist_reweighted[mc][1].SetLineColor(ROOT.TColor.GetColor(col))
    hist_reweighted[mc][1].SetFillColor(ROOT.TColor.GetColor(col))
    hist_integrals.append(hist_reweighted[mc][1].Integral())
    if hist_reweighted[mc][1].GetEntries() != 0.:
        event_weight = hist_reweighted[mc][1].Integral()/float(hist_reweighted[mc][1].GetEntries())
    else:
        event_weight = 0.0
    output = [mc, hist_reweighted[mc][1].Integral(), event_weight]
    print output
    pickle.dump(output, pickle_file_object)
    text_file_object.write(str(output)+"\n")
    hist_stack_reweighted.Add(hist_reweighted[mc][1])

    hist_original[mc][0].SetLineColor(ROOT.TColor.GetColor(col))
    hist_original[mc][0].SetFillColor(ROOT.TColor.GetColor(col))
    hist_stack_original.Add(hist_original[mc][0])

pickle_file_object.close()
text_file_object.close()

def pie_chart(hist_integrals, type):

    #mcs_legend = [r'$t(\bar{t})\gamma$', r'WZ', 'single-top', r'$t\bar{t}$', r'$V\gamma$', 'QCD', r'DY$\rightarrow ll$', r'$W \rightarrow l\nu$']
    mcs_legend = [r'$t\bar{t}$', r'$V\gamma$', 'QCD', r'DY$\rightarrow ll$', r'$W \rightarrow l\nu$']

    #plt.style.use([hep.style.ROOT, hep.style.firamath])
    fig, ax = plt.subplots()

    print hist_integrals

    lumi = ax.text(0.7, 1.1, r"%.2f fb$^{-1}$ (%s)" % (args.luminosity, args.year), fontsize=14)
    cms = ax.text(-1.6, 1.1, u"CMS $\it{Simulation Preliminary}$",fontsize=16, fontweight='bold')
    cat = ax.text(-1.6, -1.2, r"$%s$, %s, $%s$" % (category_name_latex, args.dilepton_charge, CR_var_latex), fontsize=14)

    # A standard pie plot
    patches, texts, autotexts = ax.pie(hist_integrals, labels=mcs_legend, autopct='%1.1f%%', colors=colours, shadow=True)
    plt.axis('equal')

    plt.savefig(os.path.join(args.dir,"plots/"+args.var+"_"+str(type)+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+".pdf"))
    plt.savefig(os.path.join(args.dir,"plots/"+args.var+"_"+str(type)+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+".png"))

if args.CR == "deltaR_SR":
    pie_chart(hist_integrals, "piechart_reweighted")

if hist_data is not None:
    #hist_data.Draw("P SAME")
    hist_ratio_original = hist_data.Clone("ratio histogram")
    hist_ratio_original.Divide(hist_stack_original.GetStack().Last())

    hist_ratio_reweighted = hist_data.Clone("ratio histogram")
    hist_ratio_reweighted.Divide(hist_stack_reweighted.GetStack().Last())
else:
    hist_ratio_original = define_plots_1D(binning, args.var, "Entries", "ratio_original"+key+str(random.randint(1, 1000)))
    hist_ratio_reweighted = define_plots_1D(binning, args.var, "Entries", "ratio_reweighted"+key+str(random.randint(1, 1000)))

def plotting(hist_stack, hist_data, hist_individual, hist_ratio, type):

    cv = ROOT.TCanvas()
    cv.Draw()
    cv.SetBottomMargin(0.2)
    cv.SetLeftMargin(0.13)
    #cv.SetRightMargin(0.25)

    upperPad = ROOT.TPad("upperPad", "upperPad", 0, 0.33, 1, 1)
    lowerPad = ROOT.TPad("lowerPad", "lowerPad", 0, 0, 1, 0.33)
    upperPad.SetBottomMargin(0.00001)
    upperPad.SetLeftMargin(0.15)
    upperPad.SetRightMargin(0.2)
    upperPad.SetBorderMode(0)
    upperPad.SetTopMargin(0.15)
    if args.log_scale:
        upperPad.SetLogy()
    lowerPad.SetTopMargin(0.00001)
    lowerPad.SetBottomMargin(0.4)
    lowerPad.SetLeftMargin(0.15)
    lowerPad.SetRightMargin(0.2)
    lowerPad.SetBorderMode(0)
    upperPad.Draw()
    lowerPad.Draw()

    upperPad.cd()

    hist_stack.SetMaximum(hist_stack.GetMaximum()*100)#
    hist_stack.SetMinimum(1)
    hist_stack.Draw("HIST")

    hist_stack.GetYaxis().SetTitle("Events")
    hist_stack.GetYaxis().SetTitleOffset(0.8)

    if args.CR == "deltaR_SR":
        hist_individual[args.hnl][1].Draw("SAME HIST")
        #mcs_legend = {"T*GJets*": "t(#bar{t})#gamma", "[WZ]GTo*": "V#gamma", "WZTo*": "WZ", "QCD*": "QCD", "ST*": "single-t", "TTTo*": "t#bar{t}", "DY*": "DY#rightarrowll", "W*J_*": "W#rightarrowl#nu", args.hnl: "HNL"}
        mcs_legend = {"TTTo*": "t#bar{t}", "[WZ]GTo*": "V#gamma", "QCD*": "QCD", "DY*": "DY#rightarrowll", "W*J_*": "W#rightarrowl#nu", args.hnl: "HNL"}

    if args.CR == "deltaR_CR":
        hist_data.SetLineColor(ROOT.kBlack)
        hist_data.Draw("SAME P")
        #mcs_legend = {"T*GJets*": "t(#bar{t})#gamma", "[WZ]GTo*": "V#gamma", "WZTo*": "WZ", "QCD*": "QCD", "ST*": "single-t", "TTTo*": "t#bar{t}", "DY*": "DY#rightarrowll", "W*J_*": "W#rightarrowl#nu"}
        mcs_legend = {"TTTo*": "t#bar{t}", "[WZ]GTo*": "V#gamma", "QCD*": "QCD", "DY*": "DY#rightarrowll", "W*J_*": "W#rightarrowl#nu"}


    legend = style.makeLegend(0.83,0.01,0.98,0.8)
    #legend.SetHeader(years[0], "C")
    for mc in reversed(mcs):
        if ('HNL' in mc):
            continue
        legend.AddEntry(hist_individual[mc][1], mcs_legend[mc],"f")
    #legend.AddEntry(hists_original[0], "Wjets before PU reweight","l")
    if (args.CR == "deltaR_CR"):
        legend.AddEntry(hist_data, "data","p")
    else:
        legend.AddEntry(hist_individual[args.hnl][1], mcs_legend[args.hnl],"f")
    legend.Draw("SAME")

    upperPad.Modified()

    lowerPad.cd()

    axis = hist_stack.GetStack().Last().Clone("axis")
    axis.SetMinimum(0.0)
    axis.SetMaximum(2.0)
    axis.GetXaxis().SetTitle(args.xaxis_title)
    axis.GetXaxis().SetTitleOffset(2.5)
    axis.GetYaxis().SetTitle("Data/MC")
    axis.Draw("AXIS")

    line = ROOT.TLine(0.0, 1, args.feature_max, 1)
    line.Draw("SAME")

    for ibin in range(hist_stack.GetHistogram().GetNbinsX()):
        c = hist_stack.GetStack().Last().GetBinCenter(ibin+1)
        w = hist_stack.GetStack().Last().GetBinWidth(ibin+1)
        m = hist_stack.GetStack().Last().GetBinContent(ibin+1)
        #print c, w, m
        if m > 0.0:
            h = min(hist_stack.GetStack().Last().GetBinError(ibin+1)/m, 0.399)
            #print h
            box = ROOT.TBox(c-0.5*w, 1-h, c+0.5*w, 1+h)
            box.SetFillStyle(3345)
            box.SetLineColor(ROOT.kGray+1)
            box.SetFillColor(ROOT.kGray)
            style.rootObj.append(box)
            box.Draw("SameF")
            box2 = ROOT.TBox(c-0.5*w, 1-h, c+0.5*w, 1+h)
            box2.SetFillStyle(0)
            box2.SetLineColor(ROOT.kGray+1)
            box2.SetFillColor(ROOT.kGray)
            style.rootObj.append(box2)
            box2.Draw("SameL")

    if args.CR == "deltaR_CR":
        hist_ratio.Draw("SAME P")

    cv.cd()

    style.makeLumiText(0.85, 0.97, lumi=str(args.luminosity), year=str(args.year))
    style.makeCMSText(0.13, 0.97,additionalText="Preliminary", dx=0.1)

    style.makeText(0.2, 0.80, 0.3, 0.90, category_name+", "+args.dilepton_charge+", "+CR_var)
    if args.CR == "deltaR_CR":
        style.makeText(0.15, 0.03, 0.4, 0.08, "data/MC = {0:.3g}".format(hist_data.Integral()/hist_stack.GetStack().Last().Integral()))

    cv.Modified()

    cv.SaveAs(args.dir+"plots/"+args.var+"_"+str(type)+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+".pdf")
    cv.SaveAs(args.dir+"plots/"+args.var+"_"+str(type)+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+".png")

plotting(hist_stack_reweighted, hist_data, hist_reweighted, hist_ratio_reweighted, "reweighted")
#plotting(hist_stack_original, hist_data, hist_reweighted, hist_ratio_original, "original")




def plotting_ratio(hist_ratio, type):

    cv = ROOT.TCanvas()
    cv.Draw()
    cv.SetBottomMargin(0.2)
    cv.SetLeftMargin(0.13)
    #cv.SetRightMargin(0.25)

    if args.log_scale:
        cv.SetLogy()

    hist_ratio.GetXaxis().SetTitle(args.xaxis_title)
    hist_ratio.GetYaxis().SetTitle("S/#sqrt{B}")
    hist_ratio.Draw("HIST")

    style.makeLumiText(0.85, 0.97, lumi=str(args.luminosity), year=str(args.year))
    style.makeCMSText(0.13, 0.97,additionalText="Preliminary", dx=0.1)
    style.makeText(0.2, 0.80, 0.3, 0.90, category_name+", "+args.dilepton_charge+", "+CR_var)

    cv.Modified()

    cv.SaveAs(args.dir+"plots/"+args.var+"_"+str(type)+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+".pdf")
    cv.SaveAs(args.dir+"plots/"+args.var+"_"+str(type)+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+".png")

if args.CR == "deltaR_SR":
    hist_cumulative_sig = hist_reweighted[args.hnl][1].Clone("sig vs bkg").GetCumulative(0)
    hist_cumulative_bkg = hist_stack_reweighted.GetStack().Last().GetCumulative(0)
    for bin in range(hist_cumulative_bkg.GetNbinsX()+1):
        if hist_cumulative_bkg.GetBinContent(bin+1) <= 0.:
            hist_cumulative_bkg.SetBinContent(bin+1, 1.1)
        else:
            hist_cumulative_bkg.SetBinContent(bin+1, math.sqrt(hist_cumulative_bkg.GetBinContent(bin+1)))

    hist_sig_bkg_reweighted = hist_cumulative_sig.Clone("ratio")
    hist_sig_bkg_reweighted.Divide(hist_cumulative_bkg)
    plotting_ratio(hist_sig_bkg_reweighted, "sigvsbkg_reweighted")
