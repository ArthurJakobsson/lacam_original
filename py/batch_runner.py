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

def getCSVNameFromSettings(folder, map_prefix, model):
    if model is None:
        return "{}/{}/nonn.csv".format(folder, map_prefix)
    model_prefix = model.split('/')[-1][:-3] # Remove .pt
    return "{}/{}/{}.csv".format(folder, map_prefix, model_prefix)

class BatchRunner:
    """Class for running a single scen file"""
    def __init__(self, outputcsv, mapName, scen, model, k, verbose, cutoffTime, neural) -> None:
        self.cutoffTime = cutoffTime,
        self.map = mapName
        self.scen=scen
        self.model=model
        self.k=k
        self.verbose=verbose
        self.cutoffTime=cutoffTime
        self.neural=neural
        # map_prefix = mapName.split('/')[-1][:-4]
        self.outputcsv = outputcsv

    def runSingleSettingsOnMap(self, numAgents, aSeed):
        # Main command
        command = "./build_debug/main"

        # Batch experiment settings
        command += " --seed={}".format(aSeed)
        command += " --num={}".format(numAgents)
        command += " --map={}".format(self.map)
        command += " --scen={}".format(self.scen)

        # Exp Settings
        command += " --verbose={}".format(self.verbose)
        command += " --model={}".format(self.model)
        command += " --neural_flag={}".format(self.neural)
        command += " --time_limit_sec={}".format(self.cutoffTime)
        command += " --kval={}".format(self.k)
        command += " --outputcsv={}".format(self.outputcsv)
        
        command += " --outputpaths=logs/paths.txt"

        # True if want failure error
        # print(command)
        # pdb.set_trace()
        subprocess.run(command.split(" "), check=True)

    def detectExistingStatus(self, numAgents):
        if exists(self.outputcsv):
            df = pd.read_csv(self.outputcsv)
            df = df[(df["agents"] == numAgents) & (df["scen_name"] == self.scen)]
            numFailed = len(df[(df["solved"] == '0')])
            return len(df), numFailed
        return 0, 0

    def runBatchExps(self, agentNumbers, seeds):
        for aNum in agentNumbers:
            numRan, numFailed = self.detectExistingStatus(aNum)
            if numRan > 0 and numFailed > len(seeds)/2:  # Check if existing run all failed
                print(
                    "Terminating early because all failed with {} number of agents".format(aNum))
                break
            elif numRan >= len(seeds):  # Check if ran existing run
                print("Skipping {} completely as already run!".format(aNum))
                continue
            else:
                for aSeed in seeds:
                    self.runSingleSettingsOnMap(aNum, aSeed)
                # Check if new run failed
                numRan, numFailed = self.detectExistingStatus(aNum)
                if numRan == 0:
                    raise RuntimeError("Cannot detect any runs although should have ran!")
                if numRan - numFailed <= len(seeds) / 2:
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
    for s in range(1, numScen + 1):
        myScen = scen_prefix + str(s) + ".scen"
        expSettings = dict(
            mapName=mapName,
            scen=myScen,
            model=model,
            k=k,
            verbose=1,
            cutoffTime=60,
            neural=[True, False][0],
            # output='logs/nntest_' + map_prefix + ".csv"
        )
        if expSettings["neural"] is False:
            expSettings["model"] = None

        expSettings["outputcsv"] = getCSVNameFromSettings(
            batchFolderName, map_prefix, expSettings["model"])
        
        # create directory if it does not exist
        newFolderName = "/".join(expSettings["outputcsv"].split('/')[:-1])
        if not os.path.isdir(newFolderName):
            os.makedirs(newFolderName)

        # agentRange = [1] + list(range(10, 100+1, 10))
        agentRange = list(range(50, mapsToNumAgents[map_prefix][1]+1, 50))

        seeds = list(range(1, numSeeds + 1))

        # For running across all
        myBR = BatchRunner(**expSettings)
        myBR.runBatchExps(agentRange, seeds)

"""
Example run:
python3 py/batch_runner.py --map scripts/map/random-32-32-10.map --numScen 1 --model models/random_1.pt --k 8 --numSeeds 5

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
