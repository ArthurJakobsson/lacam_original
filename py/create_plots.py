import numpy as np
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import pickle
import os
import pdb
from collections import defaultdict
from math import ceil

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plotCustomMulti(plotFunction, data, shape, filename):
    ncols,nrows = shape
    fig = plt.figure(figsize=(ncols*6+4,nrows*6))

    count = 0
    axes = []
    for x in range(nrows):
        for y in range(ncols):
            if len(data) <= count:
                continue
            ax = plt.subplot(nrows, ncols, count+1)
            axes.append(ax)
            plotFunction(ax, *data[count])
            count += 1

    xlabel = ax.xaxis.get_label().get_text()
    ylabel = ax.yaxis.get_label().get_text()
    if (nrows > 1):
        plt.setp(axes[-1, :], xlabel=xlabel)
        plt.setp(axes[:, 0], ylabel=ylabel)
    else:
        plt.setp(axes, xlabel=xlabel)
        plt.setp(axes, ylabel=ylabel)
        # if ncols == 1:
        #     plt.setp(axes, xlabel="# agents")
        #     plt.setp(axes, ylabel="Speed up")
        # else:
        #     plt.setp(axes[:], xlabel="# agents")
        #     plt.setp(axes[:], ylabel="Speed up")

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=4)
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.close('all')

# def aggStats(df, allPossibleJoinCols, aggCols):
#     # Join columns are all columns except the ones we want to aggregate over
#     joinColumns = [x for x in allPossibleJoinCols if x not in aggCols]


def loadAndCleanDf(filePath):
    def createName(row):
        if "nonn" in filePath:
            return "baseline"
        else:
            return "nn"
        # if row["solver name"] == "LNS(PP;PP)":
        #     return "LNS_S{}".format(row["sipp"])
        # return "S{}B{}PC{}C{}T{}WDG{}Sub{}".format(row["sipp"], row["bypass"], 
        #     row["prioritizingConflicts"], row["corridorReasoning"], row["targetReasoning"],
        #     row["wdgHeuristic"], row["startingSuboptimality"])

    df = pd.read_csv(filePath)
    # df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True) # Remove last column which is empty
    df['name'] = df.astype('O').apply(lambda row: createName(row), axis=1) # Name settings

    #### Aggregate over seeds and scen_name
    joinColumn = ["name", "agents", "map_file", "scen_name"]
    resultCols = [x for x in df.columns if x not in joinColumn]
    def aggStats(data):
        d = {}
        d["success"] = np.mean(data["solved"])
        data = data[data["soc"] > 0]  # Only keep successes
        for aKey in resultCols:
            if aKey in ["scen_name", "seed"]:
                continue
            if data[aKey].dtype == str:
                continue
            d[aKey] = data[aKey].median()
        # d["mid"] = d["runtime of initial solution"]
        # tmp = data[(data["solution cost"] < 1073741823) & (data["solution cost"] > 0)]
        # d["midSuccess"] = tmp["runtime of initial solution"].median()
        # d["mid"] = d["midSuccess"]
        # d["top"] = tmp["runtime of initial solution"].quantile(1)
        # d["bot"] = tmp["runtime of initial solution"].quantile(0)
        return pd.Series(d)
    dfMean = df.groupby(joinColumn, as_index=False, sort=False).apply(aggStats)
    # pdb.set_trace()
    return dfMean

def addBaseline(df, baseDf, joinColumns):
    def mergeDfs(leftDfSuffix, leftDf, rightDfSuffix, rightDf):
        df = pd.merge(leftDf, rightDf, how="inner", left_on=joinColumns, right_on=joinColumns, 
                                                    suffixes=[leftDfSuffix, rightDfSuffix])
        return df
    # pdb.set_trace()
    tmpDf = mergeDfs("", df, "_baseline", baseDf)
    # df = pd.concat([df, tmpDf], ignore_index=True, sort=False)
    # return df
    return tmpDf

def loadAndCleanDf2(filePath1, baseFilePath):
    def createName(row, filePath):
        if "nonn" in filePath:
            return "baseline"
        else:
            return "nn"
    df = pd.read_csv(filePath1)
    df['name'] = df.astype('O').apply(lambda row: createName(row, filePath1), axis=1) # Name settings
    dfBase = pd.read_csv(baseFilePath)
    dfBase['name'] = dfBase.astype('O').apply(lambda row: createName(row, baseFilePath), axis=1) # Name settings
    # pdb.set_trace()

    #### Aggregate over seeds first
    joinColumn = ["name", "agents", "map_file", "scen_name"]
    resultCols = [x for x in df.columns if x not in joinColumn]
    def aggStats(data):
        d = {}
        d["success"] = np.mean(data["solved"])
        data = data[data["soc"] > 0]  # Only keep successes
        for aKey in resultCols:
            if data[aKey].dtype == str:
                continue
            d[aKey] = data[aKey].median()
        d["stddev"] = data["soc"].std()
        return pd.Series(d)
    df = df.groupby(joinColumn, as_index=False, sort=False).apply(aggStats)
    dfBase = dfBase.groupby(joinColumn, as_index=False, sort=False).apply(aggStats)

    # pdb.set_trace()
    #### Merge with baseline
    joinColumn = ["scen_name", "map_file", "agents"] # Joining, NOT AGGREGATING, across scen_name
    resultCols = [x for x in df.columns if x not in joinColumn]

    def mergeDfs(joinColumns, leftDfSuffix, leftDf, rightDfSuffix, rightDf):
        df = pd.merge(leftDf, rightDf, how="inner", left_on=joinColumns, right_on=joinColumns, 
                                                    suffixes=[leftDfSuffix, rightDfSuffix])
        return df
    dfMerged = mergeDfs(joinColumn, "", df, "_baseline", dfBase)
    # dfMerged = dfMerged[dfMerged["success"] > 0.5]
    # pdb.set_trace()
    return dfMerged

def plotWithStddev():
    BASEFOLDER = "/home/rishi/Desktop/CMU/Research/ml-mapf/lacam/lacam_original/logs"
    SPECIFIC_FOLDER = "{}/random-32-32-10".format(BASEFOLDER)
    compareModel = ["random_1_unweight_w4full_random.csv",
                    "random_1_unweight_w4full_test_1seeds.csv"][0]
    baseModel = ["nonnfull.csv", "nonn_20seeds.csv", "nonnfull_test_20seeds.csv"][1]
    df = loadAndCleanDf2("{}/{}".format(SPECIFIC_FOLDER, compareModel),
                    "{}/{}".format(SPECIFIC_FOLDER, baseModel))

    saveFolder = SPECIFIC_FOLDER

    ### Remove failures across all agents
    numAgentsList = df["agents"].unique()
    for numAgents in numAgentsList:
        successProportion = df[df["agents"] == numAgents]["success"].mean()
        # pdb.set_trace()
        if (successProportion < 0.5): ### If aggregate fails too much, remove from df
            print("Removed {} as fails more than 50%".format(numAgents))
            df = df[df["agents"] != numAgents]
    
    print("Remaining agents: {}".format(df["agents"].unique()))
    ### Remove specific instances that fail
    df = df[df["success"] == 1]

    df["soc_dif"] = (df["soc"] - df["soc_baseline"]) / df["soc_lb"]
    # pltData = df["soc_dif"]
    # plt.boxplot(pltData.values, showfliers=False)
    # ax = plt.gca()
    # ax.set_xticklabels(pltData.keys())
    # df["tmp"] = df["stddev_baseline"] / df["soc_lb"]
    # df.boxplot(column="tmp", by="agents", grid=False, showfliers=False)
    df.boxplot(column="soc_dif", by="agents", grid=False, showfliers=False)
    plt.axhline(c='k', linestyle='--', alpha=0.5)
    plt.ylabel("(Our SoC - Baseline SoC) / LB")
    plt.savefig("testFig2".format(saveFolder), bbox_inches='tight', dpi=600)

    for numAgents in df["agents"].unique():
        tmpDf = df[df["agents"] == numAgents]
        pltData = (tmpDf["soc"].median() - tmpDf["soc_baseline"].median()) / tmpDf["soc_lb"].median()
        print("Agents: {}, median: {}".format(numAgents, pltData))

def plotInitialResults():
    BASEFOLDER = "/home/rishi/Desktop/CMU/Research/ml-mapf/lacam/lacam_original/logs"
    SPECIFIC_FOLDER = "{}/random-32-32-10".format(BASEFOLDER)
    baseDf = loadAndCleanDf("{}/random-32-32-10/nonnfull.csv".format(BASEFOLDER))
    df = loadAndCleanDf("{}/random-32-32-10/random_1_unweight_w4full_random_priority.csv".format(BASEFOLDER))
    # df = loadAndCleanDf("{}/random-32-32-10/random_1_w4_wait.csv".format(BASEFOLDER))
    # df = loadAndCleanDf("{}/random-32-32-10/random_1_sub_w4_wait.csv".format(BASEFOLDER))
    # df = pd.concat([df1, df2], ignore_index=True, sort=False)
    loadAndCleanDf2("{}/random-32-32-10/random_1_unweight_w4full_random_priority.csv".format(BASEFOLDER),
                    "{}/random-32-32-10/nonnfull.csv".format(BASEFOLDER))

    saveFolder = SPECIFIC_FOLDER

    # baseDf = df[df["name"].str.contains("baseline")]
    # pdb.set_trace()
    # df = df[df["success"] >= 0.5]
    df = addBaseline(df, baseDf, joinColumns = ["map_file", "agents", "scen_name"])

    # pdb.set_trace()
    joinColumn = ["name", "map_file", "agents"] # Joining across scen_name
    resultCols = [x for x in df.columns if x not in joinColumn]
    def aggStats(data):
        pdb.set_trace()
        d = {}
        d["success_rate"] = np.mean(data["success"])
        # if d["success_rate"] < 1:
            # pdb.set_trace()

        data = data[data["success"] > 0]  # Only keep successes
        for aKey in resultCols:
            # pdb.set_trace()
            if aKey in ["scen_name", "seed"]:
                continue
            if data[aKey].dtype != np.float64:
                continue
            d[aKey] = data[aKey].median()
        # d["mid"] = d["runtime of initial solution"]
        # tmp = data[(data["solution cost"] < 1073741823) & (data["solution cost"] > 0)]
        # d["midSuccess"] = tmp["runtime of initial solution"].median()
        # d["mid"] = d["midSuccess"]
        # d["top"] = tmp["runtime of initial solution"].quantile(1)
        # d["bot"] = tmp["runtime of initial solution"].quantile(0)
        return pd.Series(d)
    df = df.groupby(joinColumn, as_index=False, sort=False).apply(aggStats)
    # pdb.set_trace()
    df = df[df["success_rate"] > 0.5]

    df["speedup"] = df["comp_time_baseline"]/df["comp_time"]

    df["rel_path_cost"] = df["soc"]/df["soc_baseline"]
    # df["norm_soc"] = (df["soc"]-df["soc_lb"])/(df["soc_baseline"]-df["soc_lb"])
    df["norm_soc"] = (df["soc"]-df["soc_baseline"])/df["soc_lb"]
    plt.scatter(df["agents"], df["norm_soc"])
    # add line on y=0
    plt.plot([df["agents"].min(), df["agents"].max()], [0, 0], 'k-', lw=2)
    plt.ylabel("(Our SoC - Baseline SoC) / LB")
    plt.savefig("{}/path_cost_priority_random.png".format(saveFolder), bbox_inches='tight', dpi=600)
    return

    # df = df[(df["success"] >= 0.5) & (df["success_baseline"] >= 0.5)]
    df = df[df["success"] >= 0.5]

    ind = 4
    yKey = ["soc", "speedup", "success", "midSuccess"][ind]
    yLabel = ["Path cost", "Relative Speed", "Success", "Runtime (s)"][ind]
    def lsAgg(d):
        ans = {yKey: d[yKey].item()}
        # pdb.set_trace()
        ans["success"] = d["success"].item()
        return pd.Series(ans)
    ### All the columns we care about in our plots
    plotColumns = ["name", "agents"]
    lsdf = df.groupby(plotColumns, as_index=False).apply(lsAgg)
    lsdf.reset_index(drop = True, inplace = True)
    lsdf.columns = lsdf.columns.map("".join)

    ### A unique subplot based on these keys
    subPlotKeys = ["name"]
    allPlotData = []
    for label, adf in lsdf.groupby(subPlotKeys, as_index=False):
        if len(subPlotKeys) == 1:
            label = tuple([label])  # Make this a tuple
        allPlotData.append((label, adf))
    numCols = 4
    targetShape = (numCols, ceil(len(allPlotData)/numCols))

    # rename = dict(
    #     LNS_S0 = "LNS2-",
    #     LNS_S1 = "LNS2",
    #     S1B1PC0C1T1WDG0Sub5 = "EECBS",
    # )
    # off = dict(
    #     LNS_S0 = 0,
    #     LNS_S1 = -5,
    #     S1B1PC0C1T1WDG0Sub5 = 0,
    # )
    # aColor = dict(
    #     LNS_S0 = "blue",
    #     LNS_S1 = "sienna",
    #     S1B1PC0C1T1WDG0Sub5 = "green",
    # )

    # def createSubPlot(ax, label, df):
    #     dataVals = {}
    #     df = df.sort_values(['num agents', "name"], ascending=[True, True])
    #     # for index, row in df.iterrows():
    #     #     # expLabel = "w{}".format(row["w"])
    #     #     expLabel = row["name"]
    #     #     dataVals[expLabel] = row[yKey]
    #     if label[0] == 0:
    #         ax.set_title("Start-goal")
    #     else:
    #         ax.set_title("1 Start-goal-start")
    #     for nInd, aName in enumerate(df["name"].unique()):
    #         subDf = df[df["name"] == aName]
    #         print(nInd, aName)
    #         for aGroupSize in subDf["groupSize"].unique():
    #             miniDf = subDf[subDf["groupSize"] == aGroupSize]
    #             ax.plot(miniDf["num agents"], miniDf["mid"], label=rename[aName])
    #             # pdb.set_trace()
    #             ax.fill_between(miniDf["num agents"], miniDf["bot"], miniDf["top"], alpha=0.4)

    #             for i, txt in enumerate(miniDf["success"]):
    #                 offset = off[aName]
    #                 yOff = 1
    #                 if i != 0:
    #                     yOff = 1.3
    #                 ax.annotate(txt, (offset+miniDf["num agents"].iloc[i], miniDf["mid"].iloc[i]/yOff), color=aColor[aName])
    #     ax.set_ylabel(yLabel)
    #     ax.set_xlabel("Number of Agents")
    #     # ax.legend()
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=4)
    #     ax.set_yscale("log")
    #     ax.set_ylim(top=60)

    # plotCustomMulti(createSubPlot, allPlotData, targetShape, "{}/small_all{}Log5.png".format(saveFolder, yKey))

if __name__ == "__main__":
    # plotInitialResults()
    plotWithStddev()
