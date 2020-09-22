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

source /home/hep/jd918/cmsenv_setup.sh
export X509_USER_PROXY=/vols/cms/jd918/LLP/CMSSW_10_2_18/src/proxy

cd /vols/cms/jd918/LLP/CMSSW_10_2_18/src/
eval `scramv1 runtime -sh`

''')

    #submitFile.write("JOBS=(")
    for i, jobCfg in enumerate(jobArrayCfg):
        submitFile.write("if [ $SGE_TASK_ID == "+str(i+1)+" ]\n")
        submitFile.write("then\n")
        submitFile.write("python plot_validation.py "+str(jobCfg["cmd"][0])+"\n")
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

#CRs=["DY", "highMET"]
CRs=["deltaR_CR", "deltaR_SR"]
categories=["muonmuon", "muonelectron", "electronelectron", "electronmuon"]
variables={
        "EventObservables_nominal_met": [10, 0.0, 100.0, 0, "\'MET (GeV)\'", 1],
        "EventObservables_nominal_ht": [30, 0.0, 300.0, 0, "\'HT (GeV)\'", 1],
        "dilepton_mass": [20, 20.0, 80.0, 0, "\'m(l_{1}l_{2}) (GeV)\'", 1],
        "dilepton_deltaPhi": [30, -3.14, 3.14, 0, "\'#Delta#phi(l_{1}l_{2})\'", 1],
        "dilepton_deltaR": [25, 0.0, 5.0, 0, "\'#DeltaR(l_{1}l_{2})\'", 1],
        "leadingLeptons_nominal_mtw": [30, 0.0, 300.0, 1, "\'m_{T} (GeV)\'", 1],
        "leadingLeptons_nominal_deltaPhi": [30, -3.14, 3.14, 1, "\'#Delta#phi(l_{1},MET)\'", 1],
        "leadingLeptons_pt": [10, 0.0, 100.0, 1, "\'p_{T}(l_{1}) (GeV)\'", 1],
        "leadingLeptons_eta": [25, -2.5, 2.5, 1, "\'#eta(l_{1})\'", 1],
        "subleadingLeptons_pt": [10, 0.0, 100.0, 1, "\'p_{T}(l_{2}) (GeV)\'", 1],
        "subleadingLeptons_eta": [25, -2.5, 2.5, 1, "\'#eta(l_{2})\'", 1],
        "selectedJets_nominal_ptLeptonSubtracted": [10, 0.0, 100.0, 1, "\'p_{T}(jet) (GeV)\'", 1],
        "selectedJets_nominal_eta": [25, -2.5, 2.5, 1, "\'#eta(jet)\'", 1]
        # "bdt_score_nominal": [50, 0.0, 1.0, 0, "\'BDT score\'", 1],
        # "lepJet_nominal_deltaR": [55, 0.0, 5.5, 1, "\'#DeltaR(jet,l_{2})\'", 1]
        }

#luminosity={"2016": 35.92, "2017":41.53, "2018":59.74}
luminosity={"2016": 35.92}
#dir_ntuples="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/nanoAOD_friends_200622"
#dir_ntuples="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/30Jul20/"
dir_ntuples="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/19Sep20_notagger/"
oneFile = 0

for CR in CRs:
    for category in categories:
        for var, binning in variables.iteritems():
            for year, lumi in luminosity.iteritems():
                jobArrayCfg.append({
                 "cmd": [
                 "--year "+str(year)+" --CR "+str(CR)+" --category "+str(category)+" --var "+str(var)+" --luminosity "+str(lumi)+" --dir_ntuples "+str(dir_ntuples)+" --bins "+str(binning[0])+" --feature_min "+str(binning[1])+" --feature_max "+str(binning[2])+" --array_var "+str(binning[3])+" --xaxis_title "+str(binning[4])+" --log_scale "+str(binning[5])+" --oneFile "+str(oneFile)
                 ]
                })

makeSubmitFile(jobArrayCfg,"job-validation-parametric.sh")
