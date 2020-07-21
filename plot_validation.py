import sys
import os

import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import ROOT
import pandas as pd
import uproot
import random
from array import array

import subprocess
import re
import time
import json
import yaml

import style
style.makeColorTable()

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--CR', dest='CR', action='store',default="deltaR")
parser.add_argument('--category', dest='category', action='store',default="muonmuon")
parser.add_argument('--var', dest='var', action='store',default="PV_npvs")
parser.add_argument('--dir', dest='dir', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--dir_ntuples', dest='dir_ntuples', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/nanoAOD_friends_200622")
parser.add_argument('--dir_LLP', dest='dir_LLP', action='store',default="/vols/build/cms/LLP/")
parser.add_argument('--file_yaml', dest='file_yaml', action='store',default="xsec.yaml")
parser.add_argument('--dir_yields', dest='dir_yields', action='store',default="yields_200311/")
parser.add_argument('--file_yields', dest='file_yields', action='store',default="eventyields.json")
parser.add_argument('--luminosity', dest='luminosity', action='store',type=float, default=35.88)
parser.add_argument('--feature_min', dest='feature_min', action='store',type=float, default=-0.5)
parser.add_argument('--feature_max', dest='feature_max', action='store',type=float, default=60.5)
parser.add_argument('--bins', dest='bins', action='store',type=int, default=61)
parser.add_argument('--array_var', dest='array_var', action='store',type=int, default=0)

args = parser.parse_args()

category = args.category


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

def selection_deltaR(tree):
    sel = (tree.array("nleadingLepton") == 1) & \
          (tree.array("nsubleadingLepton") == 1) & \
          (tree.array("nlepJet_nominal") == 1) & \
          (tree.array("lepJet_nominal_deltaR") > 2.0) & \
          (tree.array("MET_filter") == 1) & \
          (tree.array("dilepton_mass") < 80.) & \
          (tree.array("dilepton_mass") > 20.) & \
          (tree.array("EventObservables_nominal_met") < 100.) & \
          (tree.array("dilepton_charge") == -1)#Dirac samples OS

    return sel

def dilepton_categories(category, tree):
    if category == "muonmuon":
        sel = (tree.array("category_nominal_muonmuon")==1) & \
              (tree.array("IsoMuTrigger_flag") == 1)
    elif category == "muonelectron":
        sel = (tree.array("category_nominal_muonelectron")==1) & \
              (tree.array("IsoMuTrigger_flag") == 1)
    elif category == "electronelectron":
        sel = (tree.array("category_nominal_electronelectron")==1) & \
              (tree.array("IsoElectronTrigger_flag") == 1)
    elif category == "electronmuon":
        sel = (tree.array("category_nominal_electronmuon")==1) & \
              (tree.array("IsoElectronTrigger_flag") == 1)

    return sel
#leadingLepton_isMuon, leadingLepton_isElectron IsoMuTrigger_flag IsoElectronTrigger_flag

directory_ntuples = os.path.join(args.dir_ntuples, args.year)
dirlist_ntuples = os.listdir(directory_ntuples)

xsecs_raw = yaml.load(open(os.path.join(args.dir_LLP, args.file_yaml)))
#print xsecs

yields_raw = json.load(open(os.path.join(args.dir_LLP, args.dir_yields, args.year, args.file_yields)))
#print yields

mcs = ["QCD", "ST", "TTTo", "WTo", "DY"]
#mcs = ["QCD_Pt-1000toInf"]

colours = ["#bdbdbd", "#ffb3bf", "#ef5350", "#388e3c", "#1976d2"]
files_MC = {}
files_data = {}
xsecs = {}
yields = {}

for dir in dirlist_ntuples:
    if any([mc in dir for mc in mcs]):
        if "DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-2016" in dir:
            continue
        else:
            files_MC[dir] = os.listdir(os.path.join(directory_ntuples, dir))
            for xsec_key in xsecs_raw.keys():
                if xsec_key in dir:
                    xsecs[dir] = xsecs_raw[xsec_key]
                    break
                else:
                    xsecs[dir] = xsecs_raw[xsec_key]
            for yield_key in yields_raw.keys():
                if yield_key in dir:
                    yields[dir] = yields_raw[yield_key]
                    break
    if category[0] == "m":
        if any(["SingleMuon" in dir]):
            files_data[dir] = os.listdir(os.path.join(directory_ntuples, dir))
    elif category[0] == "e":
        if any(["SingleElectron" in dir]) or any(["EGamma" in dir]):
            files_data[dir] = os.listdir(os.path.join(directory_ntuples, dir))

binning = [args.bins, args.feature_min, args.feature_max]
print binning
print(type(binning))

hists_reweighted = {key:[define_plots_1D(binning, args.var, "Entries", "samples reweighted"+key+str(i)) for i in range(3)] for key in files_MC.keys()}
hist_reweighted = {key: [define_plots_1D(binning, args.var, "Entries", "MC reweighted"+key+str(i)) for i in range(3)] for key in mcs}
hists_original = {key:[define_plots_1D(binning, args.var, "Entries", "original"+key+str(random.randint(1, 1000)))] for key in files_MC.keys()}
hist_original = {key:[define_plots_1D(binning, args.var, "Entries", "original"+key+str(random.randint(1, 1000)))] for key in mcs}
hists_data = {key:[define_plots_1D(binning, args.var, "Entries", "data"+key+str(random.randint(1, 1000)))] for key in files_data.keys()}
hist_data = define_plots_1D(binning, args.var, "Entries", "data"+str(random.randint(1, 1000)))

for dir, files in files_MC.iteritems():
    weight = eval(str(xsecs[dir])) * args.luminosity * 1e3 / yields[dir]

    print dir

    for file in files:
        #print file
        f_MC = uproot.open(os.path.join(directory_ntuples, dir, file))
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
        elif args.CR == "deltaR":
            cuts = selection_deltaR(tree)

        category_weight = dilepton_categories(args.category, tree)
        cuts = cuts*category_weight

        if args.array_var == 1:
            tree_cuts = np.array([i[0] if len(i) > 0 else None for i in tree.array(args.var)])[cuts != 0]
            tree_cuts = tree_cuts[tree_cuts != np.array(None)]
        else:
            tree_cuts = tree.array(args.var)[cuts != 0].flatten()

        genweights = tree.array("genweight")
        id_iso_weights = tree.array("tightMuon_weight_iso_nominal")*tree.array("tightMuon_weight_id_nominal")*tree.array("tightElectron_weight_id_nominal")

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

            hist_var_original = define_plots_1D(binning, args.var, "Entries", args.var+str(random.randint(1, 1000)))
            hist_var_original = plotting_1D(tree_cuts, hist_var_original, weight * all_weights)
            hists_original[dir][0] += hist_var_original

for mc in mcs:
    for dir in files_MC.keys():
        if mc in dir:
            for i, var in enumerate(["_up", "_nominal", "_down"]):
                hist_reweighted[mc][i].Add(hists_reweighted[dir][i])
                #map(lambda hist_reweighted[mc]: hist_reweighted[mc] + hists_reweighted['dir'], hists_reweighted)
            hist_original[mc][0].Add(hists_original[dir][0])

for dir, files in files_data.iteritems():

    print dir

    for file in files:
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
        elif args.CR == "deltaR":
            cuts = selection_deltaR(tree)

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

            hist_var = define_plots_1D(binning, args.var, "Entries", args.var+str(random.randint(1, 1000)))
            hist_var = plotting_1D(tree_cuts, hist_var, trigger_weight)
            hists_data[dir][0].Add(hist_var)

    hist_data.Add(hists_data[dir][0])

hist_stack_reweighted = ROOT.THStack("MC reweighted", "MC reweighted")
hist_stack_original = ROOT.THStack("MC original", "MC original")

for mc, col in zip(mcs, colours):
    hist_reweighted[mc][1].SetLineColor(ROOT.TColor.GetColor(col))
    hist_reweighted[mc][1].SetFillColor(ROOT.TColor.GetColor(col))
    print mc, hist_reweighted[mc][1].Integral()
    hist_stack_reweighted.Add(hist_reweighted[mc][1])

    hist_original[mc][0].SetLineColor(ROOT.TColor.GetColor(col))
    hist_original[mc][0].SetFillColor(ROOT.TColor.GetColor(col))
    hist_stack_original.Add(hist_original[mc][0])

if hist_data is not None:
    #hist_data.Draw("P SAME")
    hist_ratio_original = hist_data.Clone("ratio histogram")
    hist_ratio_original.Divide(hist_stack_original.GetStack().Last())

    hist_ratio_reweighted = hist_data.Clone("ratio histogram")
    hist_ratio_reweighted.Divide(hist_stack_reweighted.GetStack().Last())

#for dir, hists in hists_data.iteritems():
#    print dir, hists

def plotting(hist_stack, hist_data, hist_individual, hist_ratio, type):

    cv = ROOT.TCanvas()
    cv.Draw()
    cv.SetBottomMargin(0.2)
    cv.SetLeftMargin(0.13)

    upperPad = ROOT.TPad("upperPad", "upperPad", 0, 0.33, 1, 1)
    lowerPad = ROOT.TPad("lowerPad", "lowerPad", 0, 0, 1, 0.33)
    upperPad.SetBottomMargin(0.00001)
    upperPad.SetLeftMargin(0.15)
    upperPad.SetBorderMode(0)
    upperPad.SetTopMargin(0.15)
    upperPad.SetLogy()
    lowerPad.SetTopMargin(0.00001)
    lowerPad.SetBottomMargin(0.4)
    lowerPad.SetLeftMargin(0.15)
    lowerPad.SetBorderMode(0)
    upperPad.Draw()
    lowerPad.Draw()

    upperPad.cd()

    hist_stack.SetMaximum(hist_stack.GetMaximum()*1000)#
    hist_stack.SetMinimum(1)
    hist_stack.Draw("HIST")

    hist_stack.GetYaxis().SetTitle("Events")
    hist_stack.GetYaxis().SetTitleOffset(0.8)

    hist_data.SetLineColor(ROOT.kBlack)
    hist_data.Draw("SAME P")

    mcs_legend = {"QCD": "QCD", "ST": "ST", "TTTo": "t#bar{t}", "WTo": "W#rightarrowl#nu", "DY": "DY#rightarrowll"}

    legend = style.makeLegend(0.75,0.4,0.9,0.8)
    #legend.SetHeader(years[0], "C")
    for mc in mcs:
        legend.AddEntry(hist_individual[mc][1], mcs_legend[mc],"f")
    #legend.AddEntry(hists_original[0], "Wjets before PU reweight","l")
    legend.AddEntry(hist_data, "data","p")
    legend.Draw("SAME")

    upperPad.Modified()

    lowerPad.cd()


    axis = hist_stack.GetStack().Last().Clone("axis")
    axis.SetMinimum(0.0)
    axis.SetMaximum(2.0)
    axis.GetXaxis().SetTitle(args.var)
    axis.GetXaxis().SetTitleOffset(2.5)
    axis.GetYaxis().SetTitle("Data/MC")
    axis.Draw("AXIS")

    line = ROOT.TLine(0.0, 1, 60.0, 1)
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

    hist_ratio.Draw("SAME P")

    cv.cd()

    style.makeLumiText(0.85, 0.97, lumi=str(args.luminosity), year=str(args.year))
    style.makeCMSText(0.13, 0.97,additionalText="Preliminary", dx=0.1)

    cv.Modified()

    cv.SaveAs(args.dir+"plots/"+args.var+"_"+str(type)+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.year)+".pdf")
    cv.SaveAs(args.dir+"plots/"+args.var+"_"+str(type)+"_"+str(args.CR)+"_"+str(category)+"_"+str(args.year)+".png")

plotting(hist_stack_reweighted, hist_data, hist_reweighted, hist_ratio_reweighted, "reweighted")
plotting(hist_stack_original, hist_data, hist_reweighted, hist_ratio_original, "original")
