import statistics
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def loss_stat(file_path):
    dice_scores = []
    with open(file_path, 'r') as fp:
        lines = fp.readlines()[33:183]
        for line in lines:
            if not line.startswith("Saving"):
                line = line.strip().split(" ")
                dice_scores.append(float(line[-2]))
    return [statistics.mean(dice_scores), statistics.median(dice_scores), statistics.stdev(dice_scores),
            max(dice_scores), min(dice_scores), dice_scores]

def draw_diagram(loss_dice, dir_path):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.title("dice score trend of each loss function")
    plt.xlabel("step")
    plt.ylabel("dice score")
    for loss_func, dice_scores in loss_dice.items():
        plt.plot(list(range(len(dice_scores))), dice_scores, label=loss_func)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "dice_stat.png"))

def stat_on_all(dir_path):
    df = {"loss function": [], "mean": [], "median": [], "standard deviation": [],
          "max dice score": [], "min dice score": []}
    loss_dice = {}
    all_files = os.listdir("./AllLossTrack")
    for file in all_files:
        if file.endswith('Loss.txt'):
            stats = loss_stat(os.path.join(dir_path, file))
            df["loss function"].append(file[:-4])
            df["mean"].append(stats[0])
            df["median"].append(stats[1])
            df["standard deviation"].append((stats[2]))
            df["max dice score"].append((stats[3]))
            df["min dice score"].append((stats[4]))

            loss_dice[file[:-4]] = stats[-1]
    draw_diagram(loss_dice, dir_path)
    df = pd.DataFrame(data=df)
    df.to_csv(os.path.join(dir_path, "dice_stat.csv"), index=False)

if __name__ == "__main__":
    stat_on_all("./AllLossTrack")