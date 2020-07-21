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

    submitFile.write("JOBS=(")
    for jobCfg in jobArrayCfg:
        submitFile.write("\""+str(jobCfg["cmd"][0])+"\" ")
    submitFile.write(")\n \n")

    # for jobCfg in jobArrayCfg:
    #     submitFile.write(jobCfg["cmd"][0])
    #     submitFile.write("\n")

    submitFile.write('''echo "SGE_TASK_ID="$SGE_TASK_ID \n''')
    submitFile.write('''echo "python plot_validation.py "${JOBS[$SGE_TASK_ID-1]} \n''')
    submitFile.write("python plot_validation.py ${JOBS[$SGE_TASK_ID-1]}")

    submitFile.close()

jobArrayCfg = []

#CRs=["DY", "highMET"]
CRs=["deltaR"]
categories=["muonmuon", "muonelectron", "electronelectron", "electronmuon"]
variables={
        "PV_npvs": [61, -0.5, 60.5, 0],
        "PV_npvsGood": [61, -0.5, 60.5, 0],
        "fixedGridRhoFastjetAll": [25, 0.0, 25.0, 0],
        "nselectedJets_nominal": [11, -0.5, 10.5, 0],
        "EventObservables_nominal_met": [30, 0.0, 300.0, 0],
        "EventObservables_nominal_met_phi": [30, -3.14, 3.14, 0],
        "EventObservables_nominal_ht": [30, 0.0, 300.0, 0],
        "EventObservables_nominal_mT_met_Mu": [30, 0.0, 300.0, 0],
        "dilepton_mass": [20, 20.0, 120.0, 0],
        "dilepton_deltaPhi": [30, -3.14, 3.14, 0],
        "dilepton_deltaR": [25, 0.0, 5.0, 0],
        "leadingLepton_pt": [30, 0.0, 300.0, 1],
        "leadingLepton_eta": [25, -2.5, 2.5, 1],
        "leadingLepton_phi": [30, -3.14, 3.14, 1],
        "leadingLepton_nominal_deltaPhi": [30, -3.14, 3.14, 1],
        "subleadingLepton_pt": [30, 0.0, 300.0, 1],
        "subleadingLepton_eta": [25, -2.5, 2.5, 1],
        # "tightMuon_pt": [30, 0.0, 300.0, 1],
        # "tightMuon_eta": [25, -2.5, 2.5, 1],
        # "tightMuon_phi": [30, -3.14, 3.14, 1],
        # "looseMuons_pt": [30, 0.0, 300.0, 1],
        # "looseMuons_eta": [25, -2.5, 2.5, 1],
        # "tightElectron_pt": [30, 0.0, 300.0, 1],
        # "tightElectron_eta": [25, -2.5, 2.5, 1],
        # "tightElectron_phi": [30, -3.14, 3.14, 1],
        # "looseElectrons_pt": [30, 0.0, 300.0, 1],
        # "looseElectrons_eta": [25, -2.5, 2.5, 1],
        "selectedJets_nominal_pt": [30, 0.0, 300.0, 1],
        "selectedJets_nominal_eta": [25, -2.5, 2.5, 1],
        "bdt_score": [25, 0.0, 1.0, 0]
        }
# variables={
#             "bdt_score": [25, 0.0, 1.0, 0]
#             }

#luminosity={"2016": 35.92, "2017":41.53, "2018":59.74}
luminosity={"2016": 35.92}
#dir_ntuples="/vols/cms/jd918/LLP/CMSSW_10_2_18/src/nanoAOD_friends_200622"
dir_ntuples="/vols/cms/vc1117/LLP/nanoAOD_friends/HNL/14Jul20/"

for CR in CRs:
    for category in categories:
        for var, binning in variables.iteritems():
            for year, lumi in luminosity.iteritems():
                jobArrayCfg.append({
                 "cmd": [
                 "--year "+str(year)+" --CR "+str(CR)+" --category "+str(category)+" --var "+str(var)+" --luminosity "+str(lumi)+" --dir_ntuples "+str(dir_ntuples)+" --bins "+str(binning[0])+" --feature_min "+str(binning[1])+" --feature_max "+str(binning[2])+" --array_var "+str(binning[3])
                 ]
                })

makeSubmitFile(jobArrayCfg,"job-validation-parametric.sh")
