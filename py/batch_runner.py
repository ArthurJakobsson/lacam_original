import os
import subprocess  # For executing c++ executables
import argparse
import pdb
import pandas as pd
from os.path import exists


mapsToNumAgents = {
    "Paris_1_256": (50, 1000), # Verified
    "random-32-32-20": (50, 409), # Verified
    "random-32-32-10": (50, 461), # Verified
    "den520d": (50, 1000), # Verified
    "den312d": (50, 1000), # Verified
    "empty-32-32": (50, 511), # Verified
    "empty-48-48": (50, 1000), # Verified
    "ht_chantry": (50, 1000), # Verified
}

def getCSVNameFromSettings(folder, map_prefix, expSettings, extraSuffix=""):
    model = expSettings["model"]
    force_wait = ""
    if expSettings["force_goal_wait"]:
        force_wait = "_wait"
    
    if model is None:
        return "{}/{}{}/nonn{}.csv".format(folder, map_prefix, force_wait, extraSuffix)
    model_prefix = model.split('/')[-1][:-3] # Remove .pt
    return "{}/{}/{}{}{}.csv".format(folder, map_prefix, model_prefix, force_wait, extraSuffix)

class BatchRunner:
    """Class for running a single scen file"""
    def __init__(self, outputcsv, mapName, model, k, verbose, cutoffTime, neural, relative_last_action, 
                            target_indicator, force_goal_wait, just_pibt, tie_breaking, r_weight,
                            h_type, mult_noise) -> None:
        self.cutoffTime = cutoffTime,
        self.map = mapName
        # scen_prefix should be MapName-random- or MapName-even-, we will attach the scen number and .scen
        # assert(scen_prefix.endswith('-') and not scen_prefix.endswith('.scen'))
        # self.scen_prefix = scen_prefix
        self.model=model
        self.k=k
        self.verbose=verbose
        self.cutoffTime=cutoffTime
        self.neural=neural
        # map_prefix = mapName.split('/')[-1][:-4]
        self.outputcsv = outputcsv
        self.target_indicator = target_indicator
        self.relative_last_action = relative_last_action
        self.force_goal_wait = force_goal_wait
        self.just_pibt = just_pibt
        self.tie_breaking = tie_breaking
        self.r_weight = r_weight
        self.h_type = h_type
        self.mult_noise = mult_noise

    def runSingleSettingsOnMap(self, numAgents, aSeed, scen):
        # Main command
        command = "./build_release/main"

        # Batch experiment settings
        command += " --seed={}".format(aSeed)
        command += " --scen={}".format(scen)
        command += " --num={}".format(numAgents)
        command += " --map={}".format(self.map)

        # Exp Settings
        command += " --verbose={}".format(self.verbose)
        command += " --model={}".format(self.model)
        command += " --neural_flag={}".format(self.neural)
        command += " --time_limit_sec={}".format(self.cutoffTime)
        command += " --kval={}".format(self.k)
        command += " --force_goal_wait={}".format(self.force_goal_wait)
        command += " --outputcsv={}".format(self.outputcsv)
        
        command += " --outputpaths=logs/paths.txt"
        command += " --relative_last_action={}".format(self.relative_last_action)
        command += " --target_indicator={}".format(self.target_indicator)
        if self.r_weight == 0:
            command += " --neural_random=True" # True is better than False (verified)
        else:
            command += " --neural_random=False"
        command += " --prioritized_helpers=False"  # False is better than True (verified)
        command += " --just_pibt={}".format(self.just_pibt)
        command += " --tie_breaking={}".format(self.tie_breaking)
        command += " --r_weight={}".format(self.r_weight)
        command += " --h_type={}".format(self.h_type)
        command += " --mult_noise={}".format(self.mult_noise)

        # True if want failure error
        # print(command)
        # pdb.set_trace()
        subprocess.run(command.split(" "), check=True)

    def detectExistingStatus(self, numAgents, scens):
        if exists(self.outputcsv):
            df = pd.read_csv(self.outputcsv)
            df = df[(df["agents"] == numAgents) & df["scen_name"].isin(scens)]
            # pdb.set_trace()
            numFailed = (df["solved"] == 0).sum()
            numSuccess = (df["solved"] == 1).sum()
            assert(numFailed + numSuccess == len(df))
            return len(df), numFailed
        return 0, 0

    def runBatchExps(self, agentNumbers, seeds, scens):
        numExpPerSetting = len(seeds) * len(scens)
        for aNum in agentNumbers:
            numRan, numFailed = self.detectExistingStatus(aNum, scens)
            if numRan > 0 and numFailed > numExpPerSetting*4/5:  # Check if existing run all failed
                print(
                    "Terminating early because all failed with {} number of agents".format(aNum))
                break
            elif numRan >= numExpPerSetting:  # Check if ran existing run
                print("Skipping {} completely as already run!".format(aNum))
                continue
            else:
                ### Run across scens and seeds
                for aScen in scens:
                    for aSeed in seeds:
                        self.runSingleSettingsOnMap(aNum, aSeed, aScen)

                # Check if new run failed
                numRan, numFailed = self.detectExistingStatus(aNum, scens)
                if numRan == 0:
                    raise RuntimeError("Cannot detect any runs although should have ran!")
                # if numRan - numFailed <= numExpPerSetting/5:
                if numRan > 0 and numFailed > numExpPerSetting*4/5:
                    print(
                        "Terminating early because all failed with {} number of agents".format(aNum))
                    break


def lacamExps(mapName, numScen, model, k, numSeeds, r_weight=0, mult_noise=0, just_pibt=False):
    batchFolderName = "logs"
    # Make folder if does not exist
    if not os.path.isdir(batchFolderName):
        os.makedirs(batchFolderName)

    map_prefix = mapName.split('/')[-1][:-4]
    scen_prefix = "scripts/scen/scen-random/" + map_prefix + "-random-"
    allScens = [scen_prefix + str(s) + ".scen" for s in range(1, numScen + 1)]

    if not model.endswith(".pt"):
        if model != "None":
            raise RuntimeError("Model must be None or end in .pt")
        model = None

    # for s in range(1, numScen + 1):
        # myScen = scen_prefix + str(s) + ".scen"
    expSettings = dict(
        mapName = mapName,
        # scen = myScen,
        model = model,
        k = k,
        verbose = 1,
        cutoffTime = 60,
        neural = model is not None,
        relative_last_action = [True, False][1],
        target_indicator = [True, False][1],
        force_goal_wait = [True, False][1],
        just_pibt = just_pibt,
        tie_breaking = False,
        r_weight = r_weight,
        h_type = "perfect",
        mult_noise = mult_noise,
        # output = 'logs/nntest_' + map_prefix + ".csv"
    )
    # if expSettings["neural"] is False:
    #     expSettings["model"] = None

    if just_pibt:
        pibtString = "pibt"
    else:
        pibtString = "lacam"

    expSettings["outputcsv"] = getCSVNameFromSettings(
        # batchFolderName, map_prefix, expSettings, "_avoidAgentTie_{}_5seeds".format(pibtString))
        # batchFolderName, map_prefix, expSettings, "_manhattan_{}_5seeds".format(pibtString))
        batchFolderName, map_prefix, expSettings, "_noisy_{}_{}_5seeds".format(pibtString, int(mult_noise*100)))
        # batchFolderName, map_prefix, expSettings, "_rweightall{}_5seeds".format(int(r_weight*10)))
    
    # create directory if it does not exist
    newFolderName = "/".join(expSettings["outputcsv"].split('/')[:-1])
    if not os.path.isdir(newFolderName):
        os.makedirs(newFolderName)

    agentRange = [1] + list(range(5, 50+1, 5))
    # agentRange = list(range(50, mapsToNumAgents[map_prefix][1]+1, 50))
    # agentRange = list(range(50, 450+1, 50))

    seeds = list(range(1, numSeeds + 1))

    # For running across all
    myBR = BatchRunner(**expSettings)
    myBR.runBatchExps(agentRange, seeds, allScens)

def runRExps():
    for r_weight in [0.3, 1.5, 3, 15, 100][2:]:
        lacamExps("scripts/map/random-32-32-10.map", 25, "models/random_1_unweight_w4.pt", 
                k=4, numSeeds=5, r_weight=r_weight)
        
def runNoisyExps():
    for mult_noise in [0.03, 0.04]:
        lacamExps("scripts/map/random-32-32-10.map", 25, "None", 
                k=4, numSeeds=5, r_weight=0, mult_noise=mult_noise, just_pibt=True)
        # lacamExps("scripts/map/random-32-32-10.map", 25, "None", 
        #         k=4, numSeeds=5, r_weight=0, mult_noise=mult_noise, just_pibt=True)

"""
Example run:
python3 py/batch_runner.py --map scripts/map/random-32-32-10.map --numScen 25 \
            --model models/random_1_unweight_w4.pt --k 4 --numSeeds 5

python3 py/batch_runner.py --map scripts/map/random-32-32-10.map --numScen 25 \
            --model None --k 4 --numSeeds 5

python3 py/batch_runner.py --map scripts/map/random-32-32-10.map --numScen 25 \
            --model models/random_20_prev_action.pt --k 4 --numSeeds 5

python3 py/batch_runner.py --map scripts/map/warehouse-10-20-10-2-2.map --numScen 25 \
            --model models/warehouse.pt --k 4 --numSeeds 1
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='LaCAM batch runner',
                    description='For running experiments on different models via LaCAM',
                    epilog='Text at the bottom of help')

    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--numScen', type=int, required=True)
    parser.add_argument('--model', type=str, help="Path to model ending in .pt or [None]", required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--numSeeds', type=int, required=True)

    args = parser.parse_args()
    ## ./build/main -i scripts/scen/scen-even/den312d-even-1.scen -m scripts/map/den312d.map -v 1 --time_limit_sec=100 --neural_flag=true --model=models/den3121050agentsk8.pt -k 8 -N 1 
    lacamExps(args.map, args.numScen, args.model, args.k, args.numSeeds, just_pibt=True)
    # runRExps()
    # runNoisyExps()

    # lacamExps("scripts/map/random-32-32-10.map", 25, "None", 
    #             k=4, numSeeds=5, just_pibt=False)
    # lacamExps("scripts/map/random-32-32-10.map", 25, "None", 
    #             k=4, numSeeds=5, just_pibt=True)
