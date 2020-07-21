import os
import sys
import json
import ROOT
import math
import numpy
import scipy.interpolate

def makeLatexFile(var, CR, year):

    submitFile = open(os.path.join("png_images", str(var)+"_"+str(CR)+"_"+str(year)+".tex"),"w")#"runCombine.sh","w")
    submitFile.write('''\documentclass{article}
    \usepackage[utf8]{inputenc}

    \usepackage[landscape, margin=0.5cm, top=0.5cm, bottom=0.5cm, paperheight=25cm, paperwidth=17.6cm]{geometry}
    \usepackage{verbatim}
    \usepackage{subcaption}
    \usepackage{amssymb}
    \usepackage{multirow}
    \usepackage{hyperref}
    \usepackage{fancyhdr}
    \usepackage{float}
    \usepackage{graphicx}

    \usepackage{natbib}
    \usepackage{graphicx}

    \\begin{document}

    \pagenumbering{gobble}

    \\begin{figure}[H]

    \centering	%centres the image on the page
    \\begin{subfigure}[t]{0.48\\textwidth}
    %\hspace{0.5cm}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_muonmuon_"+str(year)+".pdf}")

    submitFile.write('''
    \end{subfigure}\hspace{0.5 cm}
    \\begin{subfigure}[t]{0.48\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_muonelectron_"+str(year)+".pdf}")

    submitFile.write('''

    \end{subfigure}

    \end{figure}

    \\begin{figure}[H]

    \\vspace{-0.5cm}

    \centering	%centres the image on the page
    \\begin{subfigure}[t]{0.48\\textwidth}
    %\hspace{0.5cm}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_electronelectron_"+str(year)+".pdf}")

    submitFile.write('''
    \end{subfigure}\hspace{0.5 cm}
    \\begin{subfigure}[t]{0.48\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_electronmuon_"+str(year)+".pdf}")

    submitFile.write('''
    \end{subfigure}

    \end{figure}

    \end{document}
    ''')

    submitFile.close()

def makeSubmitFile(jobArrayCfg, name):

    submitFile = open(name,"w")#"runCombine.sh","w")
    submitFile.write('''#!/usr/bin/bash
    #$-o png_images/output_latex_script.log

    source /home/hep/jd918/cmsenv_setup.sh
    export X509_USER_PROXY=/vols/cms/jd918/LLP/CMSSW_10_2_18/src/proxy

    cd /vols/cms/jd918/LLP/CMSSW_10_2_18/src/
    eval `scramv1 runtime -sh`

    ''')


    for jobCfg in jobArrayCfg:
        submitFile.write("pdflatex -output-directory png_images png_images/"
                        +str(jobCfg["cmd"][0])+".tex \n")
        submitFile.write("pdftoppm -png png_images/"+str(jobCfg["cmd"][0])
                        +".pdf > png_images/"+str(jobCfg["cmd"][0])+".png \n")

    submitFile.close()

categories=["muonmuon", "muonelectron", "electronelectron", "electronmuon"]


CRs = ["deltaR"]
years = ["2016"]
variables = [
            "nselectedJets_nominal", "EventObservables_nominal_met",
            "EventObservables_nominal_met_phi", "EventObservables_nominal_ht",
            "EventObservables_nominal_mT_met_Mu", "dilepton_mass",
            "dilepton_deltaPhi", "dilepton_deltaR", "leadingLepton_pt",
            "leadingLepton_eta", "leadingLepton_phi", "leadingLepton_nominal_deltaPhi",
            "subleadingLepton_pt", "subleadingLepton_eta", "selectedJets_nominal_pt",
            "selectedJets_nominal_eta", "bdt_score"
            ]

jobArrayCfg = []

for CR in CRs:
    for var in variables:
        for year in years:
            makeLatexFile(var, CR, year)
            jobArrayCfg.append({
             "cmd": [
             str(var)+"_"+str(CR)+"_"+str(year)
             ]
            })

makeSubmitFile(jobArrayCfg,"job-latex-scripts.sh")
