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
    def __init__(self, outputcsv, mapName, model, k, verbose, cutoffTime, neural, force_goal_wait) -> None:
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
        self.force_goal_wait = force_goal_wait

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
            if numRan > 0 and numFailed > numExpPerSetting/2:  # Check if existing run all failed
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
                if numRan - numFailed <= numExpPerSetting / 2:
                    print(
                        "Terminating early because all failed with {} number of agents".format(aNum))
                    break


def lacamExps(mapName, numScen, model, k, numSeeds):
    batchFolderName = "logs"
    # Make folder if does not exist
    if not os.path.isdir(batchFolderName):
        os.makedirs(batchFolderName)

    map_prefix = mapName.split('/')[-1][:-4]

    scen_prefix = "scripts/scen/scen-random/" + map_prefix + "-random-"
    allScens = [scen_prefix + str(s) + ".scen" for s in range(1, numScen + 1)]

    # for s in range(1, numScen + 1):
        # myScen = scen_prefix + str(s) + ".scen"
    expSettings = dict(
        mapName=mapName,
        # scen=myScen,
        model=model,
        k=k,
        verbose=1,
        cutoffTime=1,
        neural=[True, False][0],
        force_goal_wait=True,
        # output='logs/nntest_' + map_prefix + ".csv"
    )
    if expSettings["neural"] is False:
        expSettings["model"] = None

    expSettings["outputcsv"] = getCSVNameFromSettings(
        batchFolderName, map_prefix, expSettings, "blah")
    
    # create directory if it does not exist
    newFolderName = "/".join(expSettings["outputcsv"].split('/')[:-1])
    if not os.path.isdir(newFolderName):
        os.makedirs(newFolderName)

    # agentRange = [1] + list(range(10, 100+1, 10))
    agentRange = list(range(50, mapsToNumAgents[map_prefix][1]+1, 50))

    seeds = list(range(1, numSeeds + 1))

    # For running across all
    myBR = BatchRunner(**expSettings)
    myBR.runBatchExps(agentRange, seeds, allScens)

"""
Example run:
python3 py/batch_runner.py --map scripts/map/random-32-32-10.map --numScen 5 
            --model models/random_1_unweight_w4.pt --k 4 --numSeeds 1

"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='LaCAM batch runner',
                    description='For running experiments on different models via LaCAM',
                    epilog='Text at the bottom of help')

    parser.add_argument('--map', type=str)
    parser.add_argument( '--numScen', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--k', type=int)
    parser.add_argument('--numSeeds', type=int)

    args = parser.parse_args()
    # ./build/main -i scripts/scen/scen-even/den312d-even-1.scen -m scripts/map/den312d.map -v 1 --time_limit_sec=100 --neural_flag=true --model=models/den3121050agentsk8.pt -k 8 -N 1 
    lacamExps(args.map, args.numScen, args.model, args.k, args.numSeeds)
