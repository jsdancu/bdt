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
ROOT.TColor.InvertPalette()

#import mplhep as hep

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--year', dest='year', action='store', default="2016")
parser.add_argument('--luminosity', dest='luminosity', action='store',type=float, default=35.92)
parser.add_argument('--dir_ntuples', dest='dir_ntuples', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/nanoAOD_friends_201123")
parser.add_argument('--var', dest='var', action='store',default="bdt_score_nominal")
parser.add_argument('--var_title', dest='var_title', action='store',default="BDT score")
parser.add_argument('--feature_min', dest='feature_min', action='store',type=float, default=0.0)
parser.add_argument('--feature_max', dest='feature_max', action='store',type=float, default=1.0)
parser.add_argument('--bins', dest='bins', action='store',type=int, default=50)
parser.add_argument('--array_var', dest='array_var', action='store',type=int, default=0)
parser.add_argument('--category', dest='category', action='store',default="muonmuon")
parser.add_argument('--dilepton_charge', dest='dilepton_charge', action='store',default="SS+OS")
parser.add_argument('--dir', dest='dir', action='store',default="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/")
parser.add_argument('--dir_LLP', dest='dir_LLP', action='store',default="/vols/build/cms/LLP/")
parser.add_argument('--file_yaml', dest='file_yaml', action='store',default="xsec.yaml")
parser.add_argument('--hnl_xsec', dest='hnl_xsec', action='store',default="gridpackLookupTable.json")
parser.add_argument('--dir_yields', dest='dir_yields', action='store',default="yields_200720/")
parser.add_argument('--file_yields', dest='file_yields', action='store',default="eventyields.json")
parser.add_argument('--oneFile', dest='oneFile', action='store',type=int, default=1)

args = parser.parse_args()

category_name_dict = {}
category_name_dict["muonmuon"] = "#mu#mu"
category_name_dict["muonelectron"] = "#mue"
category_name_dict["electronmuon"] = "e#mu"
category_name_dict["electronelectron"] = "ee"
category_name = category_name_dict[args.category]

def define_plots_1D(binning1, xaxis_title, yaxis_title, title):

    hist1 = ROOT.TH1D(title, title, *binning1)

    hist1.GetXaxis().SetTitle(xaxis_title)
    hist1.GetYaxis().SetTitle(yaxis_title)
    hist1.GetXaxis().SetTitleOffset(0.8)
    hist1.GetYaxis().SetTitleOffset(0.8)

    return hist1

def define_plots_2D(binning1, binning2, xaxis_title, yaxis_title, title):

    hist1 = ROOT.TH2D(title, title, *binning1, *binning2)

    #hist1.SetTitle(title)
    hist1.GetXaxis().SetTitle(xaxis_title)
    hist1.GetYaxis().SetTitle(yaxis_title)
    hist1.GetYaxis().SetTitleOffset(0.9)

    return hist1

def plotting_1D(array1, hist1, puWeight):

    ntimes = len(array1)

    hist1.FillN(ntimes, array('d', array1), array('d', puWeight))

    return hist1

def plotting_2D(array1, array2, hist1, puWeight):

    ntimes = len(array1)

    hist1.FillN(ntimes, array('d', array1), array('d', array2), array('d', puWeight))

    return hist1

def fill_hist_1D(binning, xtitle, title, tree, weights):

    hist = define_plots_1D(binning, xtitle, "Entries", title+str(random.randint(1, 10000000)))
    hist = plotting_1D(tree, hist, weights)
    
    return hist

def fill_hist_2D(binning1, binning2, xtitle, ytitle, title, tree1, tree2, weights):

    hist = define_plots_2D(binning1, binning2, xtitle, ytitle, title+str(random.randint(1, 10000000)))
    hist = plotting_2D(tree1, tree2, hist, weights)
    
    return hist

def selection_deltaR_SR(tree):

    sel = (tree.array("nleadingLeptons") == 1) & \
          (tree.array("nsubleadingLeptons") == 1) & \
          (tree.array("nselectedJets_nominal") > 0) & \
          (tree.array("nselectedJets_nominal") < 5) & \
          (tree.array("category_simplified_nominal_index") > 0) & \
          (tree.array("MET_filter") == 1) & \
          (tree.array("dilepton_mass") < 77.) & \
          (tree.array("dilepton_mass") > 20.) & \
          (tree.array("EventObservables_nominal_met") < 100.)
          #(map(lambda x: (x < 1.3).any(), tree.array("selectedJets_nominal_minDeltaRSubtraction"))) & \

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

def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def corr_err(x, y, w, corr_coeff):
    """Uncertainty on Correlation"""
    std_x = np.std(x)
    std_y = np.std(y)
    mu_x = np.average(x, weights=w)
    mu_y = np.average(y, weights=w)

    return ((std_x*std_x + mu_x*mu_x) * (std_y*std_y + mu_y*mu_y) + 2.*corr_coeff*std_x*std_y * (2.*mu_x*mu_y + corr_coeff*std_x*std_y) - mu_x*mu_x * mu_y*mu_y)/(std_x*std_x * std_y*std_y)

'''
xsecs_raw = yaml.load(open(os.path.join(args.dir_LLP, args.file_yaml)))
yields_raw = json.load(open(os.path.join(args.dir_LLP, args.dir_yields, args.year, args.file_yields)))
'''
directory_ntuples = os.path.join(args.dir_ntuples, args.year)
dirlist_ntuples = os.listdir(directory_ntuples)

mcs = ["[WZ]GTo*", "QCD*", "DY*", "W*J_*"]
files_MC = {}
xsecs = {}
yields = {}

processes = []
for mc in mcs:
    directories = glob.glob(directory_ntuples+"/"+mc)
    print(mc, directories)
    processes.extend(directories)
print("processes: ", processes)

for dir in dirlist_ntuples:

    if any([dir in proc for proc in processes]):

        files_MC[dir] = os.listdir(os.path.join(directory_ntuples, dir))
        '''
        for xsec_key in xsecs_raw.keys():
            if xsec_key in dir:
                xsecs[dir] = xsecs_raw[xsec_key]

        for yield_key in yields_raw.keys():
            if yield_key in dir:
                yields[dir] = yields_raw[yield_key]
                break
        '''
variables =["category_simplified_nominal_llpdnnx_m_llj", args.var]

binning_mass = [25, 0.0, 200.0]
hist_mass = define_plots_1D(binning_mass, "m(l_{1}l_{2}j) (GeV)", "Entries", "mass variable")
hist_gw_mass = define_plots_1D(binning_mass, "m(l_{1}l_{2}j) (GeV)", "Entries", "mass variable (pos gw)")

binning_var = [args.bins, args.feature_min, args.feature_max]
hist_var = define_plots_1D(binning_var, variables[1], "Entries", args.var_title)
hist_gw_var = define_plots_1D(binning_var, variables[1], "Entries", args.var_title+" (pos gw)")

hist_heatmap = define_plots_2D(binning_mass, binning_var, "m(l_{1}l_{2}j) (GeV)", args.var_title, "heat map")
hist_gw_heatmap = define_plots_2D(binning_mass, binning_var, "m(l_{1}l_{2}j) (GeV)", args.var_title, "heat map (pos gw)")

tree_biggie_mass = []
tree_biggie_var = []
tree_biggie_gw_mass = []
tree_biggie_gw_var = []
all_weights_biggie = []
all_weights_biggie_gw = []

for dir, files in files_MC.items():

    #weight = eval(str(xsecs[dir])) * args.luminosity * 1e3 / yields[dir]

    if args.oneFile:
        files1 = [files[1]]
    else:
        files1 = files

    for file in files1:

        try:
            f_MC = uproot.open(os.path.join(directory_ntuples, dir, file))
        except ValueError:
            print("Could not open file.")
            continue

        if not f_MC:
            print("could not open file: ", dir, file)
            continue

        tree = f_MC["Friends"]
        if not tree:
            print("empty tree in: ", dir, file)
            continue

        cuts = selection_deltaR_SR(tree) #SR preselection cuts
        category_weight = dilepton_categories(args.category, tree) #dilepton category selection
        cuts = cuts*category_weight

        tree_cuts_mass = tree.array(variables[0])[cuts != 0].flatten()
        tree_cuts_var = tree.array(variables[1])[cuts != 0].flatten()

        if args.array_var == 1:
            tree_cuts_var = np.array([i[0] if i.size > 0 else None for i in tree.array(variables[1])])[cuts != 0]
            tree_cuts_var = tree_cuts_var[tree_cuts_var != np.array(None)]
        else:
            tree_cuts_var = tree.array(variables[1])[cuts != 0].flatten()
            
        if ("deltaPhi" in variables[1]) or ("eta" in variables[1]):
            tree_cuts_var = np.abs(tree_cuts_var)

        tree_cuts_mass = tree.array(variables[0])[cuts != 0].flatten()

        #only select events with positive genweight
        cuts_gw = cuts*(tree.array("genweight") > 0.0)
        if args.array_var == 1:
            tree_cuts_gw_var = np.array([i[0] if i.size > 0 else None for i in tree.array(variables[1])])[cuts_gw != 0]
            tree_cuts_gw_var = tree_cuts_gw_var[tree_cuts_gw_var != np.array(None)]
        else:
            tree_cuts_gw_var = tree.array(variables[1])[cuts_gw != 0].flatten()

        tree_cuts_gw_mass = tree.array(variables[0])[cuts_gw != 0].flatten()      

        genweights = tree.array("genweight")
        '''
        id_iso_weights = tree.array("tightMuon_weight_iso_nominal")*tree.array("tightMuon_weight_id_nominal")*tree.array("tightElectron_weight_id_nominal")*tree.array("tightElectron_weight_reco_nominal")*tree.array("looseElectrons_weight_reco_nominal")

        if args.category[0] == "m":
            trigger_weight = tree.array("IsoMuTrigger_weight_trigger_nominal")*tree.array("IsoMuTrigger_flag")
        elif args.category[0] == "e":
            trigger_weight = tree.array("IsoElectronTrigger_flag")

        all_weights = weight * (genweights*id_iso_weights*trigger_weight)[cuts != 0].flatten()
        all_weights_gw = weight * (genweights*id_iso_weights*trigger_weight)[cuts_gw != 0].flatten()
        '''
        all_weights = genweights[cuts != 0].flatten()
        all_weights_gw = genweights[cuts_gw != 0].flatten()

        if (len(tree_cuts_mass) > 0) and (len(tree_cuts_var) > 0):
            hist_mass.Add(fill_hist_1D(binning_mass, variables[0], "mass variable", tree_cuts_mass, all_weights))
            hist_var.Add(fill_hist_1D(binning_var, variables[1], args.var_title, tree_cuts_var, all_weights))
            hist_heatmap.Add(fill_hist_2D(binning_mass, binning_var, "m(l_{1}l_{2}j) (GeV)", args.var_title, "heat map", tree_cuts_mass, tree_cuts_var, all_weights))

        if (len(tree_cuts_gw_mass) > 0) and (len(tree_cuts_gw_var) > 0):
            hist_gw_mass.Add(fill_hist_1D(binning_mass, variables[0], "mass variable (pos gw)", tree_cuts_gw_mass, all_weights_gw))
            hist_gw_var.Add(fill_hist_1D(binning_var, variables[1], args.var_title+" (pos gw)", tree_cuts_gw_var, all_weights_gw))
            hist_gw_heatmap.Add(fill_hist_2D(binning_mass, binning_var, "m(l_{1}l_{2}j) (GeV)", args.var_title, "heat map (pos gw)", tree_cuts_gw_mass, tree_cuts_gw_var, all_weights_gw))

        tree_biggie_mass = [*tree_biggie_mass, *np.array(tree_cuts_mass)]
        tree_biggie_var = [*tree_biggie_var, *np.array(tree_cuts_var)]
        tree_biggie_gw_mass = [*tree_biggie_gw_mass, *np.array(tree_cuts_gw_mass)]
        tree_biggie_gw_var = [*tree_biggie_gw_var, *np.array(tree_cuts_gw_var)]
        all_weights_biggie = [*all_weights_biggie, *np.array(all_weights)]
        all_weights_biggie_gw = [*all_weights_biggie_gw, *np.array(all_weights_gw)]

hist_heatmap.SetMinimum(0)

text_file_object = open(os.path.join(args.dir+"plots/"+"massVar_"+args.var+"_heatmap_deltaR_SR_"+str(args.category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+'.txt'), 'w')

corr_coeff_root = hist_heatmap.GetCorrelationFactor()
print("correlation coefficient (ROOT): ", corr_coeff_root)
text_file_object.write(str(["correlation coefficient (ROOT): ", corr_coeff_root])+"\n")
corr_coeff_py = np.corrcoef(tree_biggie_mass, tree_biggie_var)[0][1]
print("correlation coefficient (Python): ", corr_coeff_py)
text_file_object.write(str(["correlation coefficient (Python): ", corr_coeff_py])+"\n")
corr_coeff_man = corr(tree_biggie_mass, tree_biggie_var, all_weights_biggie)
print("correlation coefficient (manual): ", corr_coeff_man)
text_file_object.write(str(["correlation coefficient (manual): ", corr_coeff_man])+"\n")
corr_coeff_err = corr_err(tree_biggie_mass, tree_biggie_var, all_weights_biggie, corr_coeff_man)
print("corr coeff error (manual): ", corr_coeff_err)
text_file_object.write(str(["corr coeff error (manual): ", corr_coeff_err])+"\n")

corr_coeff_root_gw = hist_gw_heatmap.GetCorrelationFactor()
print("correlation coefficient (ROOT, positive genweight): ", corr_coeff_root_gw)
text_file_object.write(str(["correlation coefficient (ROOT, positive genweight): ", corr_coeff_root_gw])+"\n")
corr_coeff_py_gw = np.corrcoef(tree_biggie_gw_mass, tree_biggie_gw_var)[0][1]
print("correlation coefficient (Python, positive genweight): ", corr_coeff_py_gw)
text_file_object.write(str(["correlation coefficient (Python, positive genweight): ", corr_coeff_py_gw])+"\n")
corr_coeff_man_gw = corr(tree_biggie_gw_mass, tree_biggie_gw_var, all_weights_biggie_gw)
print("correlation coefficient (manual, positive genweight): ", corr_coeff_man_gw)
text_file_object.write(str(["correlation coefficient (manual, positive genweight): ", corr_coeff_man_gw])+"\n")
corr_coeff_err_gw = corr_err(tree_biggie_gw_mass, tree_biggie_gw_var, all_weights_biggie_gw, corr_coeff_man_gw)
print("corr coeff error (manual, positive genweight): ", corr_coeff_err_gw)
text_file_object.write(str(["corr coeff error (manual, positive genweight): ", corr_coeff_err_gw])+"\n")

text_file_object.close()


def plotting(hist, type, corr_coeff, drawstyle="HIST"):

    cv = ROOT.TCanvas()
    cv.Draw()
    cv.SetBottomMargin(0.2)
    cv.SetLeftMargin(0.13)
    cv.SetRightMargin(0.17)

    hist.Draw(drawstyle)

    #style.makeLumiText(0.75, 0.97, lumi=str(args.luminosity), year=str(args.year))
    style.makeText(0.75, 0.93, 0.95, 0.97, "("+str(args.year)+")")
    style.makeCMSText(0.13, 0.97,additionalText="Preliminary", dx=0.1)
    style.makeText(0.2, 0.80, 0.3, 0.90, category_name+", "+args.dilepton_charge+", min(#DeltaR(l_{2}, jet))<1.3")
    if drawstyle=="COLZ":
        style.makeText(0.08, 0.07, 0.4, 0.08, "corr. coeff. (binned) = {0:.3g}".format(corr_coeff))

    cv.Modified()

    cv.SaveAs(args.dir+"plots/"+str(type)+"deltaR_SR_"+str(args.category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+".pdf")
    cv.SaveAs(args.dir+"plots/"+str(type)+"deltaR_SR_"+str(args.category)+"_"+str(args.dilepton_charge)+"_"+str(args.year)+".png")



plotting(hist_mass, "massVar_heatmap_", corr_coeff_root)
plotting(hist_var, args.var+"_heatmap_", corr_coeff_root)
plotting(hist_heatmap, "massVar_"+args.var+"_heatmap_", corr_coeff_root, drawstyle="COLZ")

plotting(hist_gw_mass, "massVar_heatmap_gw_", corr_coeff_root_gw)
plotting(hist_gw_var, args.var+"_heatmap_gw_", corr_coeff_root_gw)
plotting(hist_gw_heatmap, "massVar_"+args.var+"_heatmap_gw_", corr_coeff_root_gw, drawstyle="COLZ")
