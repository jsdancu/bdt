import os
import sys
import json
import ROOT
import math
import numpy
import scipy.interpolate

def makeLatexFile_extended(variables):

    submitFile = open(os.path.join("png_images", "all_plots_heatmap_gwonly.tex"),"w")#"runCombine.sh","w")
    submitFile.write('''\documentclass{article}
    \usepackage[utf8]{inputenc}

    \usepackage[landscape, margin=0.2cm, top=0.5cm, bottom=0.2cm, paperheight=32cm, paperwidth=13cm]{geometry}
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
    ''')

    for var, label in variables.iteritems():

        submitFile.write("\section{"+str(label)+" vs $m(l_{1}l_{2}j)$ (top all genweight, bottom +ve genweight)}")

        submitFile.write('''
        \\begin{figure}[H]

        \centering	%centres the image on the page
        \\begin{subfigure}[t]{0.24\\textwidth}
        %\hspace{0.5cm}
        ''')
        submitFile.write("\includegraphics[width =\\textwidth]{plots/"+"massVar_"+str(var)+"_heatmap_deltaR_SR_muonmuon_SS+OS_2016.pdf}")

        submitFile.write('''
        \end{subfigure}\hspace{0.3 cm}
        \\begin{subfigure}[t]{0.24\\textwidth}
        ''')

        submitFile.write("\includegraphics[width =\\textwidth]{plots/"+"massVar_"+str(var)+"_heatmap_deltaR_SR_muonelectron_SS+OS_2016.pdf}")

        submitFile.write('''

        \end{subfigure}\hspace{0.3 cm}
        \\begin{subfigure}[t]{0.24\\textwidth}
        ''')

        submitFile.write("\includegraphics[width =\\textwidth]{plots/"+"massVar_"+str(var)+"_heatmap_deltaR_SR_electronelectron_SS+OS_2016.pdf}")

        submitFile.write('''

        \end{subfigure}\hspace{0.3 cm}
        \\begin{subfigure}[t]{0.24\\textwidth}
        ''')

        submitFile.write("\includegraphics[width =\\textwidth]{plots/"+"massVar_"+str(var)+"_heatmap_deltaR_SR_electronmuon_SS+OS_2016.pdf}")

        submitFile.write('''

        \end{subfigure}

        \centering	%centres the image on the page
        \\begin{subfigure}[t]{0.24\\textwidth}
        %\hspace{0.5cm}
        ''')
        submitFile.write("\includegraphics[width =\\textwidth]{plots/"+"massVar_"+str(var)+"_heatmap_gw_deltaR_SR_muonmuon_SS+OS_2016.pdf}")

        submitFile.write('''
        \end{subfigure}\hspace{0.3 cm}
        \\begin{subfigure}[t]{0.24\\textwidth}
        ''')

        submitFile.write("\includegraphics[width =\\textwidth]{plots/"+"massVar_"+str(var)+"_heatmap_gw_deltaR_SR_muonelectron_SS+OS_2016.pdf}")

        submitFile.write('''

        \end{subfigure}\hspace{0.3 cm}
        \\begin{subfigure}[t]{0.24\\textwidth}
        ''')

        submitFile.write("\includegraphics[width =\\textwidth]{plots/"+"massVar_"+str(var)+"_heatmap_gw_deltaR_SR_electronelectron_SS+OS_2016.pdf}")

        submitFile.write('''

        \end{subfigure}\hspace{0.3 cm}
        \\begin{subfigure}[t]{0.24\\textwidth}
        ''')

        submitFile.write("\includegraphics[width =\\textwidth]{plots/"+"massVar_"+str(var)+"_heatmap_gw_deltaR_SR_electronmuon_SS+OS_2016.pdf}")

        submitFile.write('''

        \end{subfigure}

        \end{figure}
        ''')

    submitFile.write('''
    \end{document}
    ''')

    submitFile.close()

def makeSubmitFile(jobArrayCfg, jobArrayCfg_original, name):

    submitFile = open(name,"w")#"runCombine.sh","w")
    submitFile.write('''#!/usr/bin/bash
    #$-q hep.q
    #$-l h_rt=0:30:0
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

    for jobCfg in jobArrayCfg_original:
        submitFile.write("pdflatex -output-directory png_images png_images/"
                        +str(jobCfg["cmd"][0])+"_original.tex \n")
        submitFile.write("pdftoppm -png png_images/"+str(jobCfg["cmd"][0])
                        +"_original.pdf > png_images/"+str(jobCfg["cmd"][0])+"_original.png \n")

    #submitFile.write("pdflatex -output-directory png_images png_images/all_plots.tex \n")

    submitFile.close()


variables = {
            "bdt_score_nominal": "BDT score",
            "EventObservables_nominal_met": "$p^{miss}_{T}$", 
            "EventObservables_nominal_ht": "$H_{T}$",
            "dilepton_mass": "dilepton mass", 
            "dilepton_deltaPhi": "$\\Delta\\phi(l_{1}l_{2})$", 
            "dilepton_deltaR": "$\\Delta R(l_{1}l_{2})$", 
            "dilepton_charge": "dilepton charge",
            "leadingLeptons_nominal_mtw": "$m_{T}$", 
            "leadingLeptons_nominal_deltaPhi": "$\\Delta\\phi(l_{1},p^{miss}_{T})$",
            "leadingLeptons_pt": "leading lepton $p_{T}$", 
            "leadingLeptons_eta": "$\\eta(l_{1})$", 
            "subleadingLeptons_pt": "subleading lepton $p_{T}$", 
            "subleadingLeptons_eta": "$\\eta(l_{2})$", 
            "selectedJets_nominal_pt": "leading jet $p_{T}$", 
            "selectedJets_nominal_eta": "leading jet $\\eta$", 
            "leadingLeptons_isElectron": "isElectron ($l_{1}$)", 
            "subleadingLeptons_isElectron": "isElectron ($l_{2}$)"
            }

makeLatexFile_extended(variables)


