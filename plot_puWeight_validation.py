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

    sel = (tree.array("ntightMuon") == 1) & \
          (tree.array("nlooseMuons") == 1) & \
          (tree.array("MET_pt") > 100.) & \
          (tree.array("dimuon_mass") < 80.)

    return sel

def selection_DY(tree):

    sel = (tree.array("ntightMuon") == 1) & \
          (tree.array("nlooseMuons") == 1) & \
          (tree.array("dimuon_mass") > 90.) & \
          (tree.array("dimuon_mass") < 110.)

    return sel

years = ["2016", "2017", "2018"]

directory = "/vols/cms/jd918/LLP/CMSSW_10_2_18/src/"
directory_ntuples = os.path.join(directory, "nanoAOD_friends_200429", years[0])
dirlist_ntuples = os.listdir(directory_ntuples)

directory_LLP = "/vols/build/cms/LLP/"
file_yaml = "xsec.yaml"
directory_yields = "yields_200311/2016/"
file_yields = "eventyields.json"

xsecs_raw = yaml.load(open(os.path.join(directory_LLP, file_yaml)))
#print xsecs

yields_raw = json.load(open(os.path.join(directory_LLP, directory_yields, file_yields)))
#print yields

mcs = ["DY", "QCD", "ST", "TTTo", "WTo"]
#mcs = ["DY", "WTo"]

#colours = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan]
colours = ["#1976d2", "#bdbdbd", "#ffb3bf", "#ef5350", "#388e3c"]
files_MC = {}
files_data = {}
xsecs = {}
yields = {}
luminosity = 36.0e3

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
    if any(["SingleMuon" in dir]):
        files_data[dir] = os.listdir(os.path.join(directory_ntuples, dir))

#print xsecs
#print yields
# for key, value in files_MC.iteritems():
#     print key, value[:2]


"""
directory_MC = "/vols/cms/jd918/LLP/CMSSW_10_2_18/src/output_MC/"
files_MC = os.listdir(directory_MC)#"nano_100_Friend.root" #MC (Wjets) histogram
directory_data = "/vols/cms/jd918/LLP/CMSSW_10_2_18/src/output_data/"
files_data = os.listdir(directory_data)#"nano_101_Friend.root" #data histogram
"""

binning = [60, 0.0, 60.0]
#hist = define_plots_1D(binning, "# of PV in event", "Entries", "Pileup reweighting of # of PV in event")
#hists_reweighted = {key:[hist, hist, hist] for key in files_MC.keys()}
hists_reweighted = {key:[define_plots_1D(binning, "# of PV in event", "Entries", "samples reweighted"+key+str(i)) for i in range(3)] for key in files_MC.keys()}
hist_reweighted = {key: [define_plots_1D(binning, "# of PV in event", "Entries", "MC reweighted"+key+str(i)) for i in range(3)] for key in mcs}
hists_original = {key:[define_plots_1D(binning, "# of PV in event", "Entries", "original"+key+str(random.randint(1, 1000)))] for key in files_MC.keys()}
hist_original = {key:[define_plots_1D(binning, "# of PV in event", "Entries", "original"+key+str(random.randint(1, 1000)))] for key in mcs}
hists_data = {key:[define_plots_1D(binning, "# of PV in event", "Entries", "data"+key+str(random.randint(1, 1000)))] for key in files_data.keys()}
hist_data = define_plots_1D(binning, "# of PV in event", "Entries", "data"+str(random.randint(1, 1000)))

#hist_MC_integral = {key:0.0 for key in files_MC.keys()}
#MC_integral = 0.0

for dir, files in files_MC.iteritems():

    weight = xsecs[dir] * luminosity / yields[dir]

    for file in files:
        #print dir, file
        f_MC = uproot.open(os.path.join(directory_ntuples, dir, file))
        if not f_MC:
            print "could not open file: ", dir, file
            continue

        tree = f_MC["Friends"]
    	if not tree:
    	    print "empty tree in: ", dir, file
    	    continue

        #PV_npvs = tree.pandas.df(branches = ["PV_npvs", "genweight", "puweight_up", "puweight", "puweight_down", "ntightMuon", "nlooseMuons", "MET_pt", "dimuon_mass"])
        #cut_DY = "(ntightMuon == 1)*(nlooseMuons == 1)*(MET_pt > 100.)*(dimuon_mass < 80.)"

        #tree_DY = PV_npvs.query(cut_DY)
        cuts_DY = selection_highMET(tree)
        tree_DY = tree.array("PV_npvs")[cuts_DY != 0].flatten()
        genweight_DY = tree.array("genweight")[cuts_DY != 0].flatten()
        puweight_DY = tree.array("puweight")[cuts_DY != 0].flatten()
        weight_gen_pu = weight * np.multiply(genweight_DY, puweight_DY)

        # print "genweight_DY: ", len(genweight_DY)
        # print "puweight_DY: ", len(puweight_DY)
        # print "weight_gen_pu: ", len(weight_gen_pu)
        #print "len(weight * np.multiply(tree.array('genweight').flatten(), tree.array('puweight').flatten())): ", len(weight * np.multiply(tree.array("genweight").flatten(), tree.array("puweight").flatten()))

        # print "len(tree.array('PV_npvs').flatten()): ", len(tree.array('PV_npvs').flatten())
        # print "len(cuts_DY): ", len(cuts_DY)
        # print "len(tree_DY): ", len(tree_DY)
        # print "len(genweight_DY): ", len(genweight_DY)
        #print "tree.array('PV_npvs').flatten():", type(tree.array("PV_npvs").flatten())

        if len(weight_gen_pu) > 0:
            for i, var in enumerate(["_up", "", "_down"]):
                puweight_DY = tree.array("puweight"+var)[cuts_DY != 0].flatten()
                hist_PV_npvs_reweighted = define_plots_1D(binning, "# of PV in event", "Entries", "Pileup reweighting of # of PV in event"+str(random.randint(1, 1000)))
                hist_PV_npvs_reweighted = plotting_1D(tree_DY, hist_PV_npvs_reweighted, weight * np.multiply(genweight_DY, puweight_DY))
                hists_reweighted[dir][i] += hist_PV_npvs_reweighted
                #hist_PV_npvs_reweighted = plotting_1D(tree.array("PV_npvs").flatten(), hist_PV_npvs_reweighted, weight * np.multiply(tree.array("genweight").flatten(), tree.array("puweight"+var).flatten()))

            hist_PV_npvs_original = define_plots_1D(binning, "# of PV in event", "Entries", "# of PV in event"+str(random.randint(1, 1000)))
            hist_PV_npvs_original = plotting_1D(tree_DY, hist_PV_npvs_original, weight * genweight_DY)
            hists_original[dir][0] += hist_PV_npvs_original
            #hist_PV_npvs_original = plotting_1D(tree.array("PV_npvs").flatten(), hist_PV_npvs_original, weight * tree.array("genweight").flatten())


"""
    hist_MC_integral[dir] = hists_original[dir][0].Integral()

for dir, integral in hist_MC_integral.iteritems():
    print dir, integral

MC_integral = np.sum([integral for dir, integral in hist_MC_integral.iteritems()])
#print MC_integral
"""

for mc in mcs:
    for dir in files_MC.keys():
        if mc in dir:
            for i, var in enumerate(["_up", "", "_down"]):
                hist_reweighted[mc][i].Add(hists_reweighted[dir][i])
                #map(lambda hist_reweighted[mc]: hist_reweighted[mc] + hists_reweighted['dir'], hists_reweighted)
            hist_original[mc][0].Add(hists_original[dir][0])


#for mc, hist in hist_reweighted.iteritems():
#    print mc, hist


for dir, files in files_data.iteritems():
    for file in files:
        f_data = uproot.open(os.path.join(directory_ntuples, dir, file))
        tree = f_data["Friends"]

        cuts_DY = selection_highMET(tree)
        tree_DY = tree.array("PV_npvs")[cuts_DY != 0].flatten()

        if len(tree_DY) > 0:

            hist_PV_npvs = define_plots_1D(binning, "# of PV in event", "Entries", "# of PV in event"+str(random.randint(1, 1000)))
            #hist_PV_npvs = plotting_1D(tree.array("PV_npvs").flatten(), hist_PV_npvs, np.ones(len(tree.array("PV_npvs").flatten())))
            hist_PV_npvs = plotting_1D(tree_DY, hist_PV_npvs, np.ones(len(tree_DY)))
            hists_data[dir][0].Add(hist_PV_npvs)

    #scale = MC_integral/hists_data[dir][0].Integral()
    #hists_data[dir][0].Scale(scale)

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

#print hist_data

cv = ROOT.TCanvas()
cv.Draw()
cv.SetBottomMargin(0.2)
cv.SetLeftMargin(0.13)

#cv.SetLogy()
#hists_reweighted[1].SetAxisRange(0., 40.,"Y")


"""
canvas = style.makeCanvas()
canvas.SetBottomMargin(0.2)
canvas.SetTopMargin(0.1)
"""

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

hist_stack_reweighted.Draw("HIST")

hist_stack_reweighted.GetYaxis().SetTitle("Events")
hist_stack_reweighted.GetYaxis().SetTitleOffset(0.8)

hist_data.SetLineColor(ROOT.kBlack)
hist_data.Draw("SAME P")

"""
hists_reweighted[1].SetLineColor(ROOT.kBlue)
hists_reweighted[1].Draw("SAME HIST")
hists_original[0].SetLineColor(ROOT.kRed)
hists_original[0].Draw("SAME HIST")

n = hists_reweighted[1].GetNbinsX()
x = []
y = []

exl = np.zeros(n)
exh = np.zeros(n)
eyl = []
eyh = []

for i in range(hists_reweighted[1].GetNbinsX()+1):
    x.append(hists_reweighted[1].GetXaxis().GetBinCenter(i+1))
    y.append(hists_reweighted[1].GetBinContent(i+1))
    eyl.append(y[-1] - hists_reweighted[2].GetBinContent(i+1))
    eyh.append(hists_reweighted[0].GetBinContent(i+1) - y[-1])

gr = ROOT.TGraphAsymmErrors(n, array('d', x), array('d', y), exl, exh, array('d', eyl), array('d', eyh))
gr.SetMarkerStyle(8)
gr.Draw("P SAME")
"""

legend = style.makeLegend(0.75,0.45,0.95,0.75)
#legend.SetHeader(years[0], "C")
for mc in mcs:
    legend.AddEntry(hist_reweighted[mc][1], mc,"f")
#legend.AddEntry(hists_original[0], "Wjets before PU reweight","l")
legend.AddEntry(hist_data, "data","p")
legend.Draw("SAME")

upperPad.Modified()

lowerPad.cd()

axis = hist_stack_original.GetStack().Last().Clone("axis")
axis.SetMinimum(0.0)
axis.SetMaximum(2.0)
axis.GetXaxis().SetTitle("# of PV in event")
axis.GetXaxis().SetTitleOffset(2.5)
axis.GetYaxis().SetTitle("Data/MC")
axis.Draw("AXIS")

line = ROOT.TLine(0.0, 1, 60.0, 1)
line.Draw("SAME")

rootObj = []

for ibin in range(hist_stack_reweighted.GetHistogram().GetNbinsX()):
    c = hist_stack_reweighted.GetStack().Last().GetBinCenter(ibin+1)
    w = hist_stack_reweighted.GetStack().Last().GetBinWidth(ibin+1)
    m = hist_stack_reweighted.GetStack().Last().GetBinContent(ibin+1)
    #print c, w, m
    if m > 0.0:
        h = min(hist_stack_reweighted.GetStack().Last().GetBinError(ibin+1)/m, 0.399)
        #print h
        box = ROOT.TBox(c-0.5*w, 1-h, c+0.5*w, 1+h)
        box.SetFillStyle(3345)
        box.SetLineColor(ROOT.kGray+1)
        box.SetFillColor(ROOT.kGray)
        rootObj.append(box)
        box.Draw("SameF")
        box2 = ROOT.TBox(c-0.5*w, 1-h, c+0.5*w, 1+h)
        box2.SetFillStyle(0)
        box2.SetLineColor(ROOT.kGray+1)
        box2.SetFillColor(ROOT.kGray)
        rootObj.append(box2)
        box2.Draw("SameL")

hist_ratio_reweighted.Draw("SAME P")

cv.cd()
style.makeLumiText(0.8, 0.97, lumi="36.0")
style.makeCMSText(0.13, 0.97,additionalText="Simulation Preliminary")

cv.SaveAs(directory+"puWeight_validate_reweighted_highMET_2016.pdf")

upperPad.cd()

hist_stack_original.Draw("HIST")

hist_stack_original.GetYaxis().SetTitle("Events")
hist_stack_original.GetYaxis().SetTitleOffset(0.8)

hist_data.SetLineColor(ROOT.kBlack)
hist_data.Draw("SAME P")

legend.Draw("SAME")

upperPad.Modified()

lowerPad.cd()

axis.Draw("AXIS")
line.Draw("SAME")

rootObj = []

for ibin in range(hist_stack_original.GetHistogram().GetNbinsX()):
    c = hist_stack_original.GetStack().Last().GetBinCenter(ibin+1)
    w = hist_stack_original.GetStack().Last().GetBinWidth(ibin+1)
    m = hist_stack_original.GetStack().Last().GetBinContent(ibin+1)
    #print c, w, m
    if m > 0.0:
        h = min(hist_stack_original.GetStack().Last().GetBinError(ibin+1)/m, 0.399)
        #print h
        box = ROOT.TBox(c-0.5*w, 1-h, c+0.5*w, 1+h)
        box.SetFillStyle(3345)
        box.SetLineColor(ROOT.kGray+1)
        box.SetFillColor(ROOT.kGray)
        rootObj.append(box)
        box.Draw("SameF")
        box2 = ROOT.TBox(c-0.5*w, 1-h, c+0.5*w, 1+h)
        box2.SetFillStyle(0)
        box2.SetLineColor(ROOT.kGray+1)
        box2.SetFillColor(ROOT.kGray)
        rootObj.append(box2)
        box2.Draw("SameL")

hist_ratio_original.Draw("SAME P")

cv.cd()
style.makeLumiText(0.8, 0.97, lumi="36.0")
style.makeCMSText(0.13, 0.97,additionalText="Simulation Preliminary")

cv.SaveAs(directory+"puWeight_validate_original_highMET_2016.pdf")
