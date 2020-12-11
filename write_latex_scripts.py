import os
import sys
import json
import ROOT
import math
import numpy
import scipy.interpolate

def makeLatexFile(var, CR, category):

    submitFile = open(os.path.join("png_images", str(var)+"_"+str(CR)+"_"+str(category)+".tex"),"w")#"runCombine.sh","w")
    submitFile.write('''\documentclass{article}
    \usepackage[utf8]{inputenc}

    \usepackage[landscape, margin=0.5cm, top=0.5cm, bottom=0.5cm, paperheight=25cm, paperwidth=12cm]{geometry}
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
    \\begin{subfigure}[t]{0.3\\textwidth}
    %\hspace{0.5cm}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_OS_2016.pdf}")

    submitFile.write('''
    \end{subfigure}\hspace{0.3 cm}
    \\begin{subfigure}[t]{0.3\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_OS_2017.pdf}")

    submitFile.write('''

    \end{subfigure}\hspace{0.3 cm}
    \\begin{subfigure}[t]{0.3\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_OS_2018.pdf}")

    submitFile.write('''

    \end{subfigure}

    \end{figure}

    \\begin{figure}[H]

    \\vspace{-0.5cm}

    \centering	%centres the image on the page
    \\begin{subfigure}[t]{0.3\\textwidth}
    %\hspace{0.5cm}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_SS_2016.pdf}")

    submitFile.write('''
    \end{subfigure}\hspace{0.3 cm}
    \\begin{subfigure}[t]{0.3\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_SS_2017.pdf}")

    submitFile.write('''

    \end{subfigure}\hspace{0.3 cm}
    \\begin{subfigure}[t]{0.3\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_SS_2018.pdf}")

    submitFile.write('''
    \end{subfigure}

    \end{figure}

    \end{document}
    ''')

    submitFile.close()

def makeLatexFile_original(var, CR, category):

    submitFile = open(os.path.join("png_images", str(var)+"_"+str(CR)+"_"+str(category)+"_original.tex"),"w")#"runCombine.sh","w")
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
    \\begin{subfigure}[t]{0.3\\textwidth}
    %\hspace{0.5cm}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_original_"+str(CR)+"_"+category+"_OS_2016.pdf}")

    submitFile.write('''
    \end{subfigure}\hspace{0.5 cm}
    \\begin{subfigure}[t]{0.3\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_original_"+str(CR)+"_"+category+"_OS_2017.pdf}")

    submitFile.write('''

    \end{subfigure}\hspace{0.5 cm}
    \\begin{subfigure}[t]{0.3\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_original_"+str(CR)+"_"+category+"_OS_2018.pdf}")

    submitFile.write('''

    \end{subfigure}

    \end{figure}

    \\begin{figure}[H]

    \\vspace{-0.4cm}

    \centering	%centres the image on the page
    \\begin{subfigure}[t]{0.3\\textwidth}
    %\hspace{0.5cm}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_original_"+str(CR)+"_"+category+"_SS_2016.pdf}")

    submitFile.write('''
    \end{subfigure}\hspace{0.5 cm}
    \\begin{subfigure}[t]{0.3\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_original_"+str(CR)+"_"+category+"_SS_2017.pdf}")

    submitFile.write('''

    \end{subfigure}\hspace{0.5 cm}
    \\begin{subfigure}[t]{0.3\\textwidth}
    ''')

    submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_original_"+str(CR)+"_"+category+"_SS_2018.pdf}")

    submitFile.write('''
    \end{subfigure}

    \end{figure}

    \end{document}
    ''')

    submitFile.close()

def makeLatexFile_extended(variables, CRs, categories):

    submitFile = open(os.path.join("png_images", "all_plots.tex"),"w")#"runCombine.sh","w")
    submitFile.write('''\documentclass{article}
    \usepackage[utf8]{inputenc}

    \usepackage[landscape, margin=0.2cm, top=0.5cm, bottom=0.2cm, paperheight=25cm, paperwidth=13cm]{geometry}
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

    for CR in CRs:
        if CR == "deltaR_CR":
            submitFile.write("\section{Control region agreement plots}")
        else:
            submitFile.write("\section{Signal region plots}")
        for var, label in variables.iteritems():
            for category, category_label in categories.iteritems():

                #if (CR == "deltaR_SR") and (category == "muonelectron"):
                #    continue

                submitFile.write('''
                \\begin{figure}[H]

                \centering	%centres the image on the page
                \\begin{subfigure}[t]{0.3\\textwidth}
                %\hspace{0.5cm}
                ''')

                submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_OS_2016.pdf}")

                submitFile.write('''
                \end{subfigure}\hspace{0.3 cm}
                \\begin{subfigure}[t]{0.3\\textwidth}
                ''')

                submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_OS_2017.pdf}")

                submitFile.write('''

                \end{subfigure}\hspace{0.3 cm}
                \\begin{subfigure}[t]{0.3\\textwidth}
                ''')

                submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_OS_2018.pdf}")

                submitFile.write('''

                \end{subfigure}

                \centering	%centres the image on the page
                \\begin{subfigure}[t]{0.3\\textwidth}
                %\hspace{0.5cm}
                ''')

                submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_SS_2016.pdf}")

                submitFile.write('''
                \end{subfigure}\hspace{0.3 cm}
                \\begin{subfigure}[t]{0.3\\textwidth}
                ''')

                submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_SS_2017.pdf}")

                submitFile.write('''

                \end{subfigure}\hspace{0.3 cm}
                \\begin{subfigure}[t]{0.3\\textwidth}
                ''')

                submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_reweighted_"+str(CR)+"_"+category+"_SS_2018.pdf}")

                submitFile.write('''
                \end{subfigure}
                ''')
                if CR == "deltaR_CR":
                    submitFile.write("\caption*{Control region agreement plots - "+label+", "+category_label+"}")
                else:
                    submitFile.write("\caption*{Signal region plots - "+label+", "+category_label+"}")

                submitFile.write('''
                \end{figure}
                ''')

    submitFile.write("\section{Pie charts}")

    for var, label in variables.iteritems():
        for category, category_label in categories.iteritems():

            submitFile.write('''
            \\begin{figure}[H]

            \centering	%centres the image on the page
            \\begin{subfigure}[t]{0.3\\textwidth}
            %\hspace{0.5cm}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_SS+OS_2016.pdf}")

            submitFile.write('''
            \end{subfigure}\hspace{0.3 cm}
            \\begin{subfigure}[t]{0.3\\textwidth}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_SS+OS_2017.pdf}")

            submitFile.write('''

            \end{subfigure}\hspace{0.3 cm}
            \\begin{subfigure}[t]{0.3\\textwidth}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_SS+OS_2018.pdf}")

            submitFile.write('''

            \end{subfigure}

            \\newpage
            ''')

            submitFile.write("\caption*{Pie charts in Signal region - SS+OS "+label+", "+category_label+"}")
            
            submitFile.write('''
            \end{figure}
            ''')

            submitFile.write('''
            \\begin{figure}[H]

            \centering	%centres the image on the page
            \\begin{subfigure}[t]{0.3\\textwidth}
            %\hspace{0.5cm}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_OS_2016.pdf}")

            submitFile.write('''
            \end{subfigure}\hspace{0.3 cm}
            \\begin{subfigure}[t]{0.3\\textwidth}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_OS_2017.pdf}")

            submitFile.write('''

            \end{subfigure}\hspace{0.3 cm}
            \\begin{subfigure}[t]{0.3\\textwidth}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_OS_2018.pdf}")

            submitFile.write('''

            \end{subfigure}

            \centering	%centres the image on the page
            \\begin{subfigure}[t]{0.3\\textwidth}
            %\hspace{0.5cm}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_SS_2016.pdf}")

            submitFile.write('''
            \end{subfigure}\hspace{0.3 cm}
            \\begin{subfigure}[t]{0.3\\textwidth}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_SS_2017.pdf}")

            submitFile.write('''

            \end{subfigure}\hspace{0.3 cm}
            \\begin{subfigure}[t]{0.3\\textwidth}
            ''')

            submitFile.write("\includegraphics[width =\\textwidth]{plots/"+str(var)+"_piechart_reweighted_deltaR_SR_"+category+"_SS_2018.pdf}")

            submitFile.write('''
            \end{subfigure}
            ''')
            
            submitFile.write("\caption*{Pie charts in Signal region - "+label+", "+category_label+"}")
            
            submitFile.write('''
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

CRs = ["deltaR_CR", "deltaR_SR"]
categories={"muonmuon": "$\\mu \\mu$", "muonelectron": "$\\mu e$", 
            "electronelectron": "$ee$", "electronmuon": "$e \\mu$"}
years = ["2016", "2017", "2018"]
dilepton_charge=["SS", "OS"]
variables = {
            "EventObservables_nominal_met": "$p^{miss}_{T}$", 
            "EventObservables_nominal_ht": "$H_{T}$",
            "EventObservables_nominal_minPhiStar": "min($\\Delta\\phi*$)",
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
            "subleadingLeptons_isElectron": "isElectron ($l_{2}$)",
            "PV_npvs": "Number of PVs in event",
            "PV_npvsGood": "Number of PVs in event (Good)",
            "bdt_score_nominal": "BDT score"
            }
'''
variables_original = [
                    "PV_npvs", "PV_npvsGood", "fixedGridRhoFastjetAll"
                    ]
'''
makeLatexFile_extended(variables, CRs, categories)

jobArrayCfg = []

for CR in CRs:
    for var in variables.keys():
        for category in categories.keys():
            makeLatexFile(var, CR, category)
            jobArrayCfg.append({
             "cmd": [
             str(var)+"_"+str(CR)+"_"+str(category)
             ]
            })

jobArrayCfg_original = []
'''
for CR in CRs:
    for var in variables_original:
        for year in years:
            makeLatexFile_original(var, CR, year)
            jobArrayCfg_original.append({
             "cmd": [
             str(var)+"_"+str(CR)+"_"+str(year)
             ]
            })
'''
makeSubmitFile(jobArrayCfg, jobArrayCfg_original,"job-latex-scripts.sh")

