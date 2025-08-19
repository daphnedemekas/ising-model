#%%
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

#%%
results_folder = "ERconn_po_sweep1_N100"

# use the first configuration of the results to get some global constants
with open(os.path.join(results_folder, "0/results.pkl"), "rb") as fp:
    tmp = pickle.load(fp)
    N, n_trials = tmp["N"], len(tmp["avg_vfe_per_trial"])

p_list = []
po_list = []
for cond_folder_i in os.listdir(results_folder):
    res_file_name = os.path.join(results_folder, cond_folder_i, "results.pkl")
    if os.path.exists(res_file_name):
        with open(res_file_name, "rb") as fp:
            tmp = pickle.load(fp)
            p, po = tmp["p"], tmp["po"]
            p_list.append(p)
            po_list.append(po)

# starting_p = 1.5 * np.log(N)/N # log(N)/N is the value of p where network is connected about 36% of the time
# ER_p_levels = np.linspace( starting_p, 1.0, 20)
# po_levels = np.linspace( 0.5 + 1e-16, 1.0 - 1e-16, 50)

ER_p_levels = np.unique(np.array(p_list))
po_levels = np.unique(np.array(po_list))
#%%
num_dep_measures = 5

measure_stats_per_condition = np.empty(
    (len(ER_p_levels), len(po_levels), num_dep_measures, 3)
)

for cond_folder_i in os.listdir(results_folder):

    res_file_name = os.path.join(results_folder, cond_folder_i, "results.pkl")
    if os.path.exists(res_file_name):
        with open(res_file_name, "rb") as fp:
            tmp = pickle.load(fp)
            p, po = tmp["p"], tmp["po"]

            ii = np.where(ER_p_levels == p)[0][0]
            jj = np.where(po_levels == po)[0][0]

            measure_stats_per_condition[ii, jj, 0, 0] = np.nanmean(
                tmp["avg_vfe_per_trial"]
            )
            measure_stats_per_condition[ii, jj, 1, 0] = np.nanmean(
                tmp["avg_complexity_per_trial"]
            )
            measure_stats_per_condition[ii, jj, 2, 0] = np.nanmean(
                tmp["avg_neg_accur_per_trial"]
            )
            measure_stats_per_condition[ii, jj, 3, 0] = np.nanmean(
                tmp["avg_polarization_per_trial"]
            )
            measure_stats_per_condition[ii, jj, 4, 0] = np.nanmean(
                tmp["avg_m_per_trial"]
            )

            measure_stats_per_condition[ii, jj, 0, 1] = np.nanstd(
                tmp["avg_vfe_per_trial"]
            )
            measure_stats_per_condition[ii, jj, 1, 1] = np.nanstd(
                tmp["avg_complexity_per_trial"]
            )
            measure_stats_per_condition[ii, jj, 2, 1] = np.nanstd(
                tmp["avg_neg_accur_per_trial"]
            )
            measure_stats_per_condition[ii, jj, 3, 1] = np.nanstd(
                tmp["avg_polarization_per_trial"]
            )
            measure_stats_per_condition[ii, jj, 4, 1] = np.nanstd(
                tmp["avg_m_per_trial"]
            )

            measure_stats_per_condition[ii, jj, 0, 2] = np.count_nonzero(
                np.isnan(tmp["avg_vfe_per_trial"])
            )
            measure_stats_per_condition[ii, jj, 1, 2] = np.count_nonzero(
                np.isnan(tmp["avg_complexity_per_trial"])
            )
            measure_stats_per_condition[ii, jj, 2, 2] = np.count_nonzero(
                np.isnan(tmp["avg_neg_accur_per_trial"])
            )
            measure_stats_per_condition[ii, jj, 3, 2] = np.count_nonzero(
                np.isnan(tmp["avg_polarization_per_trial"])
            )
            measure_stats_per_condition[ii, jj, 4, 2] = np.count_nonzero(
                np.isnan(tmp["avg_m_per_trial"])
            )

#%%

# measure_idx_mapping = {0: 'Average VFE', 1: 'Average complexity', 2: 'Average neg. accuracy', 3: 'Average polarization', 4:'Average branching parameter'}
measure_idx_mapping = {
    0: "Average VFE",
    1: "Average complexity",
    2: "Average neg. accuracy",
    4: "Average branching parameter",
}

plot_loglog = False

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(16, 12))
for ER_p_idx, ER_p in enumerate(ER_p_levels):

    subplot_counter = 0
    for measure_idx, measure_name in measure_idx_mapping.items():
        mean_val = measure_stats_per_condition[ER_p_idx, :, measure_idx, 0]
        std_val = measure_stats_per_condition[ER_p_idx, :, measure_idx, 1]
        num_nans = measure_stats_per_condition[ER_p_idx, :, measure_idx, 2]
        use_std = True
        if use_std:
            sems = std_val
        else:
            sems = std_val / (np.sqrt(n_trials - num_nans))

        nan_idx = np.where(num_nans >= (0.01 * n_trials))[0]

        if len(nan_idx) > 0:
            up_to = nan_idx[0]
        else:
            up_to = len(po_levels)

        if measure_name == "Average VFE" or measure_name == "Average neg. accuracy":
            up_to = 40

        if plot_loglog:

            axes.flatten()[subplot_counter].loglog(
                po_levels[:up_to],
                mean_val[:up_to],
                lw=1,
                label="$p = $%.2f" % (ER_p_levels[ER_p_idx]),
            )
            if (
                measure_name == "Average complexity"
                or measure_name == "Average branching parameter"
            ):
                axes.flatten()[subplot_counter].set_ylim(1e-1, 1e0)
            else:
                axes.flatten()[subplot_counter].set_ylim(1e-1, 1e2)
        else:
            axes.flatten()[subplot_counter].fill_between(
                po_levels[:up_to],
                mean_val[:up_to] + sems[:up_to],
                mean_val[:up_to] - sems[:up_to],
                alpha=0.4,
            )
            axes.flatten()[subplot_counter].plot(
                po_levels[:up_to],
                mean_val[:up_to],
                lw=1,
                label="$p = $%.2f" % (ER_p_levels[ER_p_idx]),
            )

        axes.flatten()[subplot_counter].set_xlabel("$p_{\mathcal{O}}$", fontsize=18)

        if subplot_counter == 1:
            axes.flatten()[subplot_counter].legend(
                loc=(1.01, 0.01), ncol=4, fontsize=20
            )
        axes.flatten()[subplot_counter].set_title(
            "%s as $p_{\mathcal{O}}$" % (measure_name), fontsize=20
        )

        subplot_counter += 1

if plot_loglog:
    fig.savefig("ERconn_vs_po_sweep1loglog.pdf", dpi=325, bbox_inches="tight")
else:
    fig.savefig("ERconn_vs_po_sweep1.pdf", dpi=325, bbox_inches="tight")

# %%
