import os
import sys
import json
import ROOT
import math
import numpy as np
import scipy.interpolate
import itertools
import json

# variables_perm = [
#                     "EventObservables_nominal_ht", "dilepton_mass", "dilepton_deltaPhi", 
#                     "dilepton_deltaR", "leadingLeptons_pt", "subleadingLeptons_pt", 
#                     "selectedJets_nominal_pt", "subselectedJets_nominal_pt"
#                 ]
variables_perm = [
                    "EventObservables_nominal_ht", "leadingLeptons_pt", "subleadingLeptons_pt", 
                    "selectedJets_nominal_pt", "subselectedJets_nominal_pt"
                ]
variables_const = [
                    "EventObservables_nominal_met", "leadingLeptons_nominal_mtw", 
                    "leadingLeptons_nominal_deltaPhi", "leadingLeptons_eta", "subleadingLeptons_eta", 
                    "selectedJets_nominal_eta", "subselectedJets_nominal_eta", "dilepton_charge",
                    "leadingLeptons_isElectron", "subleadingLeptons_isElectron"
                    ]

dict_combinations = {}
index = 0
#dict_combinations[0] = variables_perm+variables_const
textFile = open("bdt/bdt_inputs_hyperparam.txt","w") 

for n_comb in range(4):
    variables = list(itertools.combinations(variables_perm, len(variables_perm)-n_comb))
    print("list variables: ", variables)
    print("len list: ",len(variables))
    for i, var in enumerate(variables):
        print(list(var)+variables_const)
        dict_combinations[i+index] = list(var)+variables_const
        textFile.write(str(list(var)+variables_const)+"\n")
        print(i+index)

    index+=len(variables)

textFile.close()   

with open("bdt/bdt_inputs_hyperparam.json","w") as jsonFile:
    json.dump(dict_combinations, jsonFile, indent=1)

# result = itertools.combinations([1, 2, 3], 2)
# print(*result)