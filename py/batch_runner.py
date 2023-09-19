import os
import subprocess  # For executing c++ executables
import argparse
import pdb
import pandas as pd
from os.path import exists


class BatchRunner:
    """Class for running a single scen file"""

    def __init__(self, mapName, scen, model, k, verbose, cutoffTime, neural, output) -> None:
        self.cutoffTime = cutoffTime,
        self.map = mapName
        self.scen=scen
        self.model=model
        self.k=k
        self.verbose=verbose
        self.cutoffTime=cutoffTime
        self.neural=neural
        self.output = output
        map_prefix = mapName.split('/')[-1][:-4]
        self.outputCSVFile = "logs/nntest_" + map_prefix + ".csv"

    def runSingleSettingsOnMap(self, numAgents, aSeed):
        # Main command
        command = "./build/main"

        # Batch experiment settings
        command += " --seed={}".format(aSeed)
        command += " --num={}".format(numAgents)
        command += " --map=" + self.map
        command += " --scen=" + str(self.scen)
        command += " --model=" + str(self.model)

        # Exp Settings
        command += " --verbose={}".format(self.verbose)
        command += " --neural_flag={}".format(self.neural)
        command += " --time_limit_sec={}".format(self.cutoffTime)
        command += " --kval={}".format(self.k)
        command += " --output={}".format(self.output)

        # True if want failure error
        subprocess.run(command.split(" "), check=True)

    def detectExistingStatus(self, numAgents):
        if exists(self.outputCSVFile):
            df = pd.read_csv(self.outputCSVFile)
            df = df[(df["agents"] == numAgents) & (df["scen_name"] == self.scen)]
            numFailed = len(df[(df["solved"] == '0')])
            return len(df), numFailed
        return 0, 0

    def runBatchExps(self, agentNumbers, seeds):
        for aNum in agentNumbers:
            numRan, numFailed = self.detectExistingStatus(aNum)
            if numFailed > len(seeds)/2:  # Check if existing run all failed
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
                if numRan - numFailed <= len(seeds) / 2:
                    print(
                        "Terminating early because all failed with {} number of agents".format(aNum))
                    break


def lacamExps(mapName, numScen, model, k, numSeeds):
    batchFolderName = "logs/"
    # Make folder if does not exist
    if not os.path.isdir(batchFolderName):
        os.makedirs(batchFolderName)

    map_prefix = mapName.split('/')[-1][:-4]

    scen_prefix = "scripts/scen/scen-even/" + map_prefix + "-even-"
    for s in range(1, numScen + 1):
        myScen = scen_prefix + str(s) + ".scen"
        expSettings = dict(
            mapName=mapName,
            scen=myScen,
            model=model,
            k=k,
            verbose=1,
            cutoffTime=60,
            neural="true",
            output='logs/nntest_' + map_prefix + ".csv"
        )

        agentRange = [1] + list(range(10, 100+1, 10))

        seeds = list(range(1, numSeeds + 1))

        # For running across all
        myBR = BatchRunner(**expSettings)
        myBR.runBatchExps(agentRange, seeds)


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
