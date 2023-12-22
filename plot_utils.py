import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import scipy
import statsmodels.api as sm
from datetime import datetime
from pathlib import Path
import os
import sys
import argparse
import logging

# Set the global font size for labels and titles
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14 
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12


COL_DATASET = "dataset"
COL_BASEDIST = "base_dist"
COL_NUM_LAYERS = "num_layers"
COL_NUM_UNITS = "num_units"
COL_LOGN = "$\\log n$"
COL_LAMLOGN = "$-\\lambda \\log n$"
COL_H = "$H$"
COL_ELBO_S = "ELBO+$nS_n$"
COL_DIMW = "$dim_w$"
COL_DIMQ = "$dim_q$"
COL_LAMBDA = "$\\lambda$"
COL_SEED = "seed"
COL_ELBO_SHAT = "ELBO+$n\\hat S_n$"
COL_TESTLPD = "test_lpd"
COL_LR = "lr"
COL_MVFE = "normalized MVFE"
COL_VGE = "VGE"
COL_LINESCORE = "line_score"

BASE_DISTRIBUTIONS = ["gengamma", "gaussian"]
INDEX_COLS = [COL_DATASET, COL_BASEDIST, COL_NUM_LAYERS, COL_NUM_UNITS]

SLOPE_VAR_NAME_DICT = {
    COL_MVFE: "$\\hat{\\lambda}_{vfe}$",
    COL_VGE: "$\\hat{\\lambda}_{vge}$",
}

FILENAME_PREFIX_DICT = {
    COL_ELBO_S: "ELBO_nS_vs_logn",
    COL_ELBO_SHAT: "ELBO_nShat_vs_logn",
    COL_TESTLPD: f"{COL_TESTLPD}_vs_logn",
    COL_MVFE: "MVFE_vs_logn",
    COL_VGE: "VGE_vs_inverse_n",
}


def argparser():
    """
    Parse commandline arguments into Namespace() object.
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--datafilepaths", type=str, nargs="*", required=True)
    parser.add_argument(
        "--output_dirpath",
        required=False,
        type=str,
        default=os.curdir,
        help="path to output directory. Default: current directory",
    )
    parser.add_argument(
        "--logfilename",
        metavar="FILENAME",
        type=str,
        default="stdout",
        help="Logging file. Default: stdout",
    )
    return parser


def start_log(logfilepath, level=logging.INFO):
    """
    Initiate program logging.
    """
    if logfilepath == "stdout":
        logstream = sys.stdout
    else:
        logstream = open(logfilepath, "w")
    logging.basicConfig(
        stream=logstream,
        level=level,
        filemode="w",
        format="%(asctime)s %(message)s",
        datefmt="[%m/%d/%Y %H:%M:%S] ",
    )
    logging.info("Program started")
    logging.info("Command line: {0}\n".format(" ".join(sys.argv)))
    return


def parse_raw_dataframe(datafilepaths):
    df_raw = pd.concat([pd.read_pickle(filepath) for filepath in datafilepaths])
    rec = []
    for method in df_raw["method"]:
        s = [entry.strip("\\") for entry in method.split("_")]
        num_layers, layer_width = s[-2:]
        m = s[0]
        rec.append([m, int(num_layers), int(layer_width)])

    df_raw[["base_dist", COL_NUM_LAYERS, COL_NUM_UNITS]] = rec
    return df_raw


def filter_dataframe(df_raw):
    groupby_cols = INDEX_COLS + [COL_H, COL_LOGN, COL_LR]
    df = None
    for _, g in df_raw.groupby(groupby_cols):
        mean = np.mean(g[COL_MVFE])
        filter_flag = np.abs(g[COL_MVFE] - mean) / mean <= 0.8
        if df is None:
            df = g[filter_flag]
        else:
            df = pd.concat([df, g[filter_flag]])

    return df.set_index(INDEX_COLS)


def compute_line_score(df_group):
    groupby_cols = INDEX_COLS + [COL_H]
    group = df_group.groupby(groupby_cols)
    rec = {}
    for name, g in group:
        y = g[(COL_MVFE, "mean")]
        line_score = np.sum(np.abs(y))
        rec[name] = line_score

    x = []
    for idx, row in df_group.iterrows():
        key = tuple(list(idx) + [row[COL_H].item()])
        x.append(rec[key])
    return x


def _transpose_list(x):
    return list(map(list, zip(*x)))


def plot_experiments(
    plot_grid,
    df_plot,
    plot_stats="mean",
    plot_least_square_line=True,
):

    basedist_color_map = {"gengamma": "Greens", "gaussian": "Reds"}
    basedist_marker_map = {"gengamma": "x", "gaussian": "^"}

    num_rows = len(plot_grid)
    num_cols = len(plot_grid[0])
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows), sharex=False
    )
    axes = np.array(axes)
    if len(axes.shape) <= 1:
        axes = axes.reshape(-1, 1)

    rec = []
    for i in range(num_rows):
        for j in range(num_cols):
            dataset, h, plot_yvar = plot_grid[i][j]
            ax = axes[i][j]
            df = df_plot.loc[dataset]
            df = df[df[COL_H] == h]
            complexities = sorted(
                set(
                    zip(
                        df.reset_index()[COL_NUM_LAYERS],
                        df.reset_index()[COL_NUM_UNITS],
                    )
                )
            )
            for idx in df.index.unique():
                base_dist, n, m = idx
                # Pulling out relevant columns / quantities
                x = df.loc[idx, COL_LOGN]
                xlabel = COL_LOGN
                if plot_yvar in [COL_TESTLPD, COL_VGE]:
                    x = 1 / np.exp(x)
                    xlabel = "1/n"

                y = df.loc[idx, (plot_yvar, plot_stats)]
                y2 = df.loc[idx, (COL_LAMLOGN, plot_stats)]

                #             e = df.loc[idx, (PLOT_YVAR, "std")]
                e_min = df.loc[idx, (plot_yvar, "min")]
                e_max = df.loc[idx, (plot_yvar, "max")]
                e = np.abs(np.array([e_min, e_max]) - np.array([y]))

                lmbda = df[(COL_LAMBDA, plot_stats)]
                cmap = matplotlib.colormaps[basedist_color_map[base_dist]]
                c = cmap((complexities.index((n, m)) + 1) / len(complexities))
                marker = basedist_marker_map[base_dist]
                if plot_least_square_line:
                    linefit_result = sm.OLS(np.array(y), np.array(x)).fit()
                    if plot_yvar == COL_VGE:
                        slope = linefit_result.params[0]
                        intercept = 0
                        rsquared_value = linefit_result.rsquared
                    else:
                        (
                            slope,
                            intercept,
                            r_value,
                            p_value,
                            std_err,
                        ) = scipy.stats.linregress(x, y)
                        rsquared_value = r_value**2

                    ax.plot(
                        x, intercept + slope * x, linestyle="dashed", color=c, alpha=0.8
                    )
                    slope_var_name = SLOPE_VAR_NAME_DICT.get(plot_yvar, "slope")
                    base_dist_str = 'N' if base_dist=='gaussian' else '$\\gamma$'
                    label = (
                        f"{base_dist_str}_{n}_{m},"
                        f" $R^2$={np.around(rsquared_value, 2)},"
                        f" {slope_var_name}={np.around(slope, 2)}"
                    )    
                else:
                    label = f"{base_dist_str}_{n}_{m}"
                    slope = intercept = rsquared_value = None

                rec.append(
                    [
                        dataset,
                        base_dist,
                        n,
                        m,
                        h,
                        slope,
                        intercept,
                        rsquared_value,
                        lmbda[0],
                    ]
                )
                ax.errorbar(
                    x,
                    y,
                    fmt=f"{marker}-",
                    yerr=e,
                    capsize=5,
                    elinewidth=1,
                    capthick=1,
                    label=label,
                    color=c,
                    markersize=8
                )

                ax.legend()
                ax.set_title(f"{dataset}, H={h}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(plot_yvar)
                if plot_yvar == COL_VGE:
                    ax.ticklabel_format(
                        axis="x", style="scientific", scilimits=(0, 0), useMathText=True
                    )
    return fig, axes, rec


def generate_plot_config(
    df_group,
    plot_base_dist,
    plot_best_line_only,
    plot_var,
):
    dataset_types = sorted(set([x[0] for x in df_group.index]))
    plot_grid = []
    for dataset in dataset_types:
        df = df_group.loc[dataset]
        row = []
        for h in sorted(df[COL_H].unique()):
            row.append((dataset, h, plot_var))
        plot_grid.append(row)

    idx_filtered = df_group.index.get_level_values(INDEX_COLS.index(COL_BASEDIST)).isin(
        plot_base_dist
    )
    df_plot = df_group[idx_filtered]

    if plot_best_line_only:
        df_chosen = None
        groupby_cols = ["dataset", "base_dist", COL_H]
        for _, df in df_plot.groupby(groupby_cols):
            chosen_rows = df[df["line_score"] == df["line_score"].min()]
            if df_chosen is None:
                df_chosen = chosen_rows
            else:
                df_chosen = pd.concat([df_chosen, chosen_rows])
        df_plot = df_chosen
    return plot_grid, df_plot


def plot_lambdas_comparisons(recs, rsquared_thresh=0.9):
    df1 = recs[COL_MVFE][[COL_BASEDIST, COL_NUM_LAYERS, COL_NUM_UNITS, SLOPE_VAR_NAME_DICT[COL_MVFE], "$R^2$-value"]]
    df2 = recs[COL_VGE][[COL_BASEDIST, COL_NUM_LAYERS, COL_NUM_UNITS, SLOPE_VAR_NAME_DICT[COL_VGE], "$R^2$-value"]]
    idx_cols = [COL_DATASET, COL_H, COL_BASEDIST,  COL_NUM_LAYERS, COL_NUM_UNITS]
    df1 = df1.reset_index().set_index(idx_cols)
    df2 = df2.reset_index().set_index(idx_cols)
    df = df1.join(df2, lsuffix="_vfe", rsuffix="_vge")
    df["vi_architecture"] = ['_'.join(map(str, x[2:])) for x in df.index]
    df_p = df[(df["$R^2$-value_vfe"] > rsquared_thresh) & (df["$R^2$-value_vge"] > rsquared_thresh)]
    

    fig_lambda_scatter, ax = plt.subplots(figsize=(8, 8))
    xvar_name = SLOPE_VAR_NAME_DICT[COL_MVFE]
    yvar_name = SLOPE_VAR_NAME_DICT[COL_VGE]
    sns.scatterplot(
        data=df_p,
        x=xvar_name, 
        y=yvar_name, 
        hue=COL_BASEDIST, 
        style=COL_DATASET, 
        legend="auto", 
        ax=ax, 
        s=100
    )
    xmax = ax.get_xlim()[1]
    ymax = ax.get_ylim()[1]
    xrange = np.linspace(0, xmax, num=50)
    ax.plot(xrange, xrange, "b--", alpha=0.5)
    ax.set_ylim(0, ymax)


    fig_lambda_boxplot, ax = plt.subplots(figsize=(8, 8))
    sns.boxplot(
        data=df_p.reset_index(), 
        x="dataset", 
        y=SLOPE_VAR_NAME_DICT[COL_VGE], 
        hue="base_dist", 
        ax=ax, 
        color="white", 
    )
    sns.stripplot(
        data=df_p.reset_index(), 
        x="dataset", 
        y=SLOPE_VAR_NAME_DICT[COL_VGE], 
        hue="base_dist", 
        dodge=True,
        ax=ax
    )
    ax.set_yscale("log")


    fig_vge_compare, ax = plt.subplots(figsize=(8, 8))
    a = {}
    for index, row in df_p.iterrows():
        dset, h, bdist, nl, nu = index
        key = '_'.join(map(str, [dset, h, nl, nu]))
        if key not in a:
            a[key] = {}
        a[key][bdist] = row[SLOPE_VAR_NAME_DICT[COL_VGE]]
    dfx = pd.DataFrame.from_dict(a, orient="index")
    dfx = dfx.dropna(axis=0)
    ax.plot(
        dfx["gaussian"], 
        dfx["gengamma"], 
        "kx"
    )
    ax.set_xlabel("Gaussian base dist $\\hat{\\lambda}_{vge}$")
    ax.set_ylabel("Gengamma base dist $\\hat{\\lambda}_{vge}$")
    xmax = ax.get_xlim()[1]
    xrange = np.linspace(0, xmax, num=50)
    ax.plot(xrange, xrange, "b--", alpha=0.5)
    
    return fig_lambda_scatter, fig_lambda_boxplot, fig_vge_compare

def main():
    # parse commandline arguments
    commandline_args = argparser().parse_args()
    OUTPUT_DIR = commandline_args.output_dirpath
    if not os.path.isdir(OUTPUT_DIR):
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    start_log(commandline_args.logfilename)
    logging.info(f"Commandline arguments:\n{commandline_args}")

    def savefig(fig, filename):
        filepath = os.path.join(OUTPUT_DIR, filename)
        logging.info(f"Saving figure: {filepath}")
        fig.savefig(filepath, bbox_inches="tight")
        return

    df_raw = parse_raw_dataframe(commandline_args.datafilepaths)
    df_data = filter_dataframe(df_raw)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    ax = axes[0]
    ax.hist(df_raw[COL_MVFE])
    ax.set_yscale("log")
    ax.set_title(f"Before filtering, nrows: {df_raw.shape[0]}")

    ax = axes[1]
    ax.hist(df_data[COL_MVFE])
    ax.set_yscale("log")
    ax.set_title(f"After filtering, nrows: {df_data.shape[0]}")
    savefig(fig, "MVFE_histogram_before_and_after_filtering.png")
    df_data.drop(["method", "grad_flag"], axis=1, inplace=True)
    # print(df_data["method"])
    # return
    df_group = (
        df_data.reset_index()
        .groupby(INDEX_COLS + [COL_LR, COL_H, COL_LOGN])
        .agg(["mean", "std", "min", "median", "max"])
        .reset_index()
        .set_index(INDEX_COLS)
    )
    df_group[COL_LINESCORE] = compute_line_score(df_group)

    # Plot....
    recs = {}
    for base_dist_choice in [BASE_DISTRIBUTIONS] + [
        [distname] for distname in BASE_DISTRIBUTIONS
    ]:
        for plot_var in [COL_MVFE, COL_VGE]:
            plot_grid, df_plot = generate_plot_config(
                df_group,
                plot_base_dist=base_dist_choice,
                plot_best_line_only=False,
                plot_var=plot_var,
            )
            fig, axes, rec = plot_experiments(plot_grid, df_plot, plot_least_square_line=True)
            image_filename = (
                FILENAME_PREFIX_DICT.get(plot_var, None)
                + f"_{'_'.join(base_dist_choice)}"
                + ".png"
            )
            
            if base_dist_choice == BASE_DISTRIBUTIONS: # only record one where all base dist were represented
                recs[plot_var] = pd.DataFrame(
                    rec, 
                    columns=[
                    COL_DATASET, COL_BASEDIST, COL_NUM_LAYERS, 
                    COL_NUM_UNITS, COL_H, SLOPE_VAR_NAME_DICT[plot_var], 
                    "intercept", "$R^2$-value", "$\\lambda$"
                    ]
                ).set_index([COL_DATASET, COL_H])
            savefig(fig, image_filename)
    
    fig_lambda_scatter, fig_lambda_boxplot, fig_vge_compare = plot_lambdas_comparisons(recs, rsquared_thresh=0.9)
    savefig(fig_lambda_scatter, "lambda_vfe_vge_compare.png")
    savefig(fig_lambda_boxplot, "lambda_vge_boxplot_basedist_compare.png")
    savefig(fig_vge_compare, "lambda_vge_scatter_basedist_compare.png")
    # Plot best line only
    for plot_var in [COL_MVFE, COL_VGE]:
        plot_grid, df_plot = generate_plot_config(
            df_group,
            plot_base_dist=BASE_DISTRIBUTIONS,
            plot_best_line_only=True,
            plot_var=plot_var,
        )
        fig, axes, rec = plot_experiments(plot_grid, df_plot, plot_least_square_line=True)
        image_filename = (
            FILENAME_PREFIX_DICT.get(plot_var, None)
            + f"_{'_'.join(BASE_DISTRIBUTIONS)}"
            + "_best_line_only.png"
        )
        savefig(fig, image_filename)

    # Figure in main paper.
    plot_grid = [
        [("ffrelu", 40, COL_MVFE), ("ffrelu", 40, COL_VGE)],
        [("reducedrank", 16, COL_MVFE), ("reducedrank", 16, COL_VGE)],
        [("tanh", 280, COL_MVFE), ("tanh", 280, COL_VGE)],
    ]
    plot_grid = _transpose_list(plot_grid)
    _, df_plot = generate_plot_config(
        df_group,
        plot_base_dist=BASE_DISTRIBUTIONS,
        plot_best_line_only=False,
        plot_var=COL_MVFE,  # this parameter is ignored since we are hardcoding the plot_grid this time.
    )
    fig, axes, rec = plot_experiments(plot_grid, df_plot, plot_least_square_line=True)
    # image_filename = "experiment_result_plots_vertical.png"
    image_filename = "experiment_result_plots_horizontal.png"
    savefig(fig, image_filename)

    # Plotting only architecture with the least complexity
    least_complexity = sorted(
        set(
            zip(
                df_plot.index.get_level_values(COL_NUM_LAYERS),
                df_plot.index.get_level_values(COL_NUM_UNITS),
            )
        )
    )[0]
    for plot_var in [COL_MVFE, COL_VGE]:
        plot_grid, df_plot = generate_plot_config(
            df_group,
            plot_base_dist=BASE_DISTRIBUTIONS,
            plot_best_line_only=False,
            plot_var=plot_var,
        )
        mask = (
            df_plot.index.get_level_values(COL_NUM_LAYERS) == least_complexity[0]
        ) & (df_plot.index.get_level_values(COL_NUM_UNITS) == least_complexity[1])
        df_plot = df_plot[mask]
        fig, axes, rec = plot_experiments(plot_grid, df_plot, plot_least_square_line=True)
        image_filename = (
            FILENAME_PREFIX_DICT.get(plot_var, None)
            + f"_{'_'.join(BASE_DISTRIBUTIONS)}"
            + f"_{least_complexity[0]}_{least_complexity[1]}_only.png"
        )
        savefig(fig, image_filename)
    return


if __name__ == "__main__":
    main()
