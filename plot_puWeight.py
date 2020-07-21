import sys
import os

import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import ROOT
import pandas as pd
import uproot
from array import array

import subprocess
import re
import time
import json

def define_plots_1D(binning1, xaxis_title, yaxis_title, title):

    hist1 = ROOT.TH1D(title, title, *binning1)

    hist1.GetXaxis().SetTitle(xaxis_title)
    hist1.GetYaxis().SetTitle(yaxis_title)
    hist1.GetYaxis().SetTitleOffset(0.9)

    return hist1

def plotting_1D(array1, hist1):

    ntimes = len(array1)
    w = array('d', np.ones(ntimes))

    hist1.FillN(ntimes, array('d', array1), w)

    return hist1

directory = "/vols/cms/jd918/LLP/CMSSW_10_2_18/src/"
file = "nano_10_Friend.root"

f = uproot.open(directory + file)
keys = f.keys()

binning = [40, 0.0, 2.0]
hists = []

for key in keys:
    tree = f[key]
    #print tree.array("puweight").flatten()

    for var in ["_up", "", "_down"]:
        hist_puweight = define_plots_1D(binning, "PU weight", "Entries", "PU weight")
        hist_puweight = plotting_1D(tree.array("puweight"+var).flatten(), hist_puweight)
        hists.append(hist_puweight)

cv = ROOT.TCanvas()
cv.Draw()

hists[0].SetLineColor(ROOT.kBlue)
hists[0].Draw()
hists[1].SetLineColor(ROOT.kRed)
hists[1].Draw("SAME")
hists[2].SetLineColor(ROOT.kMagenta)
hists[2].Draw("SAME")

legend = ROOT.TLegend(0.6,0.5,0.9,0.7)
legend.AddEntry(hists[0], "puWeight_up", "l")
legend.AddEntry(hists[1], "puWeight", "l")
legend.AddEntry(hists[2], "puWeight_down", "l")
legend.Draw("SAME")

cv.SaveAs(directory+"puWeight.pdf")
