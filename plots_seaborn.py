import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dir = 'seaborn_figures/proj/'
sns.set_theme()


def plot():
    # read by default 1st sheet of an excel file
    sheets = ["bp", "reverse", "sr", "mmd"]
    y_ax = ["Rewards", "Reverse KL", "Success Rate", "MMD"]
    #sheets = ["l2", "SE"]
    #y_ax = ["L2 Distance", "Samples"]
    shareys = [True, False, False, False, True, False]
    for i, c in enumerate(sheets):
        dataframe1 = pd.read_excel('paper_data.xlsx', sheet_name=sheets[i])
        g = sns.catplot(x="Trajectories", y=y_ax[i],
                        hue="Algorithm", col="Environment",
                        data=dataframe1, kind="bar",
                        height=4, aspect=.7, sharey=shareys[i], legend=False)

        (g.set_titles("{col_name}")).despine(left=True)

        """plt.xticks(
            rotation=45,
            horizontalalignment='right',
            fontweight='light',
            fontsize='xx-small'
        )"""
        plt.subplots_adjust(left=0.09, bottom=0.15)
        plt.savefig(dir + sheets[i] + '.pdf')


def plot_proj():
    # read by default 1st sheet of an excel file
    #sheets = ["best_proj", "l2_proj", "se_proj", "sr_proj"]
    #y_ax = ["Rewards", "L2 Distance", "Samples", "Success Rate"]
    sheets = ["l2_proj", "se_proj", "sr_proj"]
    y_ax = ["L2 Distance", "Samples", "Success Rate"]
    #sheets = ["rkl_proj"]
    #y_ax = ["Reverse KL"]

    for i, c in enumerate(sheets):
        dataframe1 = pd.read_excel('paper_data.xlsx', sheet_name=sheets[i])
        # Draw a nested barplot by species and sex
        g = sns.catplot(x="Environment", y=y_ax[i],
                        hue=" ", data=dataframe1, kind="bar",
                        legend_out=False, legend=False)
        #plt.ylim(0, 600)
        plt.xticks(rotation=45,
                   horizontalalignment='right',
                   fontweight='light',
                   )
        g.set(xlabel="")
        plt.subplots_adjust(left=0.15, bottom=0.21)
        plt.savefig(dir + sheets[i] + ".pdf")

if __name__ == "__main__":
    plot()
