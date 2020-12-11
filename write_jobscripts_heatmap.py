import os
import sys
import json
import ROOT
import math
import numpy
import scipy.interpolate

def makeSubmitFile(jobArrayCfg,name):

    submitFile = open(name,"w")#"runCombine.sh","w")
    submitFile.write('''#!/usr/bin/bash
#$ -cwd
#$ -q hep.q
#$ -l h_rt=03:00:00
#$ -j y
#$ -t 1-'''+str(len(jobArrayCfg))+'''
#$ -o job_logfiles/output_$JOB_ID_$TASK_ID.log

eval "$(/home/hep/jd918/anaconda3/bin/conda shell.bash hook)"
conda activate HNL

cd /vols/cms/jd918/LLP/CMSSW_10_2_18/src/

''')

    #submitFile.write("JOBS=(")
    for i, jobCfg in enumerate(jobArrayCfg):
        submitFile.write("if [ $SGE_TASK_ID == "+str(i+1)+" ]\n")
        submitFile.write("then\n")
        submitFile.write("python heatmaps.py "+str(jobCfg["cmd"][0])+"\n")
        submitFile.write("fi\n")
    #submitFile.write(")\n \n")

    # for jobCfg in jobArrayCfg:
    #     submitFile.write(jobCfg["cmd"][0])
    #     submitFile.write("\n")

    # submitFile.write('''echo "SGE_TASK_ID="$SGE_TASK_ID \n''')
    # submitFile.write('''echo "python plot_validation.py "${JOBS[$SGE_TASK_ID-1]} \n''')
    # submitFile.write("python plot_validation.py \"${${JOBS[$SGE_TASK_ID-1]}[@]}\"")

    submitFile.close()

jobArrayCfg = []


CRs=["deltaR_SR"]

categories=["muonmuon", "muonelectron", "electronelectron", "electronmuon"]

dilepton_charge=["SS+OS"]

variables={
        "EventObservables_nominal_met": [20, 0.0, 100.0, 0, "\'p^{miss}_{T} (GeV)\'"],
        "EventObservables_nominal_ht": [30, 0.0, 300.0, 0, "\'H_{T} (GeV)\'"],
        "dilepton_mass": [20, 20.0, 80.0, 0, "\'m(l_{1}l_{2}) (GeV)\'", 1],
        "dilepton_deltaPhi": [20, -0.14, 3.14, 0, "\'#Delta#phi(l_{1}l_{2})\'"],
        "dilepton_deltaR": [25, 0.0, 5.0, 0, "\'#DeltaR(l_{1}l_{2})\'"],
        "leadingLeptons_nominal_mtw": [30, 0.0, 300.0, 1, "\'m_{T} (GeV)\'"],
        "leadingLeptons_nominal_deltaPhi": [20, -0.14, 3.14, 1, "\'#Delta#phi(l_{1},p^{miss}_{T})\'"],
        "leadingLeptons_pt": [20, 25.0, 100.0, 1, "\'p_{T}(l_{1}) (GeV)\'"],
        "leadingLeptons_eta": [20, -0.4, 2.4, 1, "\'#eta(l_{1})\'"],
        "subleadingLeptons_pt": [20, 3.0, 100.0, 1, "\'p_{T}(l_{2}) (GeV)\'"],
        "subleadingLeptons_eta": [20, -0.4, 2.4, 1, "\'#eta(l_{2})\'"],
        "selectedJets_nominal_pt": [20, 15.0, 100.0, 1, "\'p_{T}(leading jet) (GeV)\'"],
        "selectedJets_nominal_eta": [20, -0.4, 2.4, 1, "\'#eta(leading jet)\'"],
        "leadingLeptons_isElectron": [3, -0.25, 1.25, 1, "\'isElectron (l_{1})\'"],
        "subleadingLeptons_isElectron": [3, -0.25, 1.25, 1, "\'isElectron (l_{2})\'"],
        "dilepton_charge": [3, -1.5, 1.5, 0, "\'sign(l_{1}#timesl_{2})\'"],
        "bdt_score_nominal": [50, 0.0, 1.0, 0, "\'BDT score\'"]
        #"EventObservables_nominal_minPhiStar": [30, 0, 3.14, 0, "\'min(#Delta#phi*)\'", 1]
        #"bdt_score_nominal": [50, 0.0, 1.0, 0, "\'BDT score\'", 1],
        # "lepJet_nominal_deltaR": [55, 0.0, 5.5, 1, "\'#DeltaR(jet,l_{2})\'", 1]
        }

#luminosity={"2016": 35.92, "2017":41.53, "2018":59.68}
luminosity={"2016": 35.92}

#dir_ntuples="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/19Sep20_notagger"
dir_ntuples="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/nanoAOD_friends_201123"
oneFile = 0


for category in categories:
    for dilep_ch in dilepton_charge:
        for var, binning in variables.items():
            for year, lumi in luminosity.items():
                jobArrayCfg.append({
                    "cmd": [
                    "--year "+str(year)+" --category "+str(category)+" --dilepton_charge "+str(dilep_ch)+" --var "+str(var)+" --luminosity "+str(lumi)+" --dir_ntuples "+str(dir_ntuples)+" --bins "+str(binning[0])+" --feature_min "+str(binning[1])+" --feature_max "+str(binning[2])+" --array_var "+str(binning[3])+" --var_title "+str(binning[4])+" --oneFile "+str(oneFile)
                    ]
                })

makeSubmitFile(jobArrayCfg,"job-heatmap.sh")
