import os
import subprocess  # For executing c++ executables
import argparse
import pdb
import pandas as pd
from os.path import exists


class BatchRunner:
    """Class for running a single scen file"""

    def __init__(self, mapName, scen, model, k, verbose, cutoffTime, neural, output, df) -> None:
        self.cutoffTime = cutoffTime
        self.map = mapName
        self.scen=scen,
        self.model=model,
        self.k=k,
        self.verbose=verbose,
        self.cutoffTime=cutoffTime,
        self.neural=neural,
        self.output = output
        self.outputCSVFile = "./csv/"+mapName

    def runSingleSettingsOnMap(self, numAgents):
        # Main command
        command = "./build/main"

        # Batch experiment settings
        # command += " --seed={}".format(aSeed)
        command += " --num={}".format(numAgents)
        command += " --map={}",format(self.map)
        command += " --scen=".format(self.scen)
        command += " --model=".format(self.model)

        # Exp Settings
        command += " --verbose=".format(self.verbose)
        command += " --neural_flag={}".format(self.neural)
        command += " --time_limit_sec={}".format(self.cutoffTime)
        command += " --kval=".format(self.k)
        command += " --output=".format(self.output)

        # True if want failure error
        subprocess.run(command.split(" "), check=True)
        f = open("./logs.txt", "r")
        agents = f.readline().split("=")[1]
        map_file = f.readline().split("=")[1]
        next(f) #skip solver
        solved = f.readline().split("=")[1]
        soc = f.readline().split("=")[1]
        soc_lb = f.readline().split("=")[1]
        makespan = f.readline().split("=")[1]
        makespan_lb = f.readline().split("=")[1]
        sum_of_loss = f.readline().split("=")[1]
        sum_of_loss_lb = f.readline().split("=")[1]
        comp_time = f.readline().split("=")[1]
        dfList= [map_file, self.scen, self.k, self.neural, agents,  solved, soc, soc_lb, makespan, makespan_lb, sum_of_loss, sum_of_loss_lb, comp_time]
        new_df = pd.DataFrame(dfList).T
        new_df.to_csv('./csv/'+self.map+'.csv', mode='a')

        # convert to csv here

    def detectExistingStatus(self, numAgents):
        

        if exists(self.outputCSVFile):
            df = pd.read_csv(self.outputCSVFile)
            if self.solver == "LNS":
                solverName = "LNS(PP;PP)"
            else:
                solverName = "AnytimeEECBS"
            # pdb.set_trace()
            df = df[(df["solver name"] == solverName) & (df["num agents"] == numAgents) & (df["sipp"] == self.sipp)
                    & (df["cutoffTime"] == self.cutoffTime) & (df["goBack"] == self.goBack) & (df["groupSize"] == self.groupSize)]
            if self.eecbsParams is not None:
                # ep = self.eecbsParams
                for aKey, aVal in self.eecbsParams.items():
                    df = df[(df[aKey] == aVal)]
                # df[(df["bypass"] == ep["bypass"]) & (df["prioritizingc"] == ep["prioritizingConflicts"]]
            numFailed = len(df[(df["solution cost"] <= 0) |
                            (df["solution cost"] >= 1073741823)])
            return len(df), numFailed
        return 0, 0

    def runBatchExps(self, agentNumbers, seeds):
        for aNum in agentNumbers:
            numRan, numFailed = self.detectExistingStatus(aNum)
            if numFailed >= len(seeds)/2:  # Check if existing run all failed
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


def lacamExps(map, scen, model, k):
    batchFolderName = "logs/"

    df = pd.DataFrame(columns=['map', 'scene','k','neural','agents','solved', 'soc', 'soc_lb', 'makespan','makespan_lb', 'sum_of_loss', 'sum_of_loss_lb', 'comp_time'])

    df.to_csv('./csv/'+map+'.csv')

    expSettings = dict(
        map=map,
        scen=scen,
        model=model,
        k=k,
        verbose=True,
        cutoffTime=60,
        neural=True,
    )
    
    # Make folder if does not exist
    if not os.path.isdir(batchFolderName):
        os.makedirs(batchFolderName)

    agentRange = [1] + list(range(10, 100+1, 10))

    # seeds = list(range(1, 3))

    # For running across all
    myBR = BatchRunner(**expSettings)
    myBR.runBatchExps(agentRange)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='LaCAM batch runner',
                    description='For running experiments on different models via LaCAM',
                    epilog='Text at the bottom of help')

    parser.add_argument('-m', '--map')
    parser.add_argument('-s', '--scen')
    parser.add_argument('-md', '--model')
    parser.add_argument('-k', '--k')

    args = parser.parse_args()
    
    lacamExps(args.map, args.scen, args.model, args.k)
