import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import seaborn as sns
from matplotlib import colormaps


def plot_trees_in_x_y_plane(
    ground_truth_trees,
    estimated_trees,
    sample_z_distance,
    median_dist,
    mean_dist,
    std_dist,
    output_path,
    title,
    model_index,
):
    gt_x = ground_truth_trees["x"]
    gt_y = ground_truth_trees["y"]

    est_x = estimated_trees["x"]
    est_y = estimated_trees["y"]

    plt.figure(figsize=(10, 10))

    ax = plt.axes([0.1, 0.1, 0.7, 0.7])
    sc = ax.scatter(est_x, est_y, c=sample_z_distance, cmap="viridis", s=20)
    ax.scatter(gt_x, gt_y, c="red", marker="x", s=30)

    plt.title(f"Trees Positions ({title})", fontsize=18)

    ax.set_xlabel("x [m]", fontsize=14, labelpad=10)
    ax.set_ylabel("y [m]", fontsize=14, labelpad=10)
    legend = ax.legend(["estimated", "ground truth"], fontsize=14, loc="upper left")
    ax.set_xlim(min(gt_x) - 5, max(gt_x) + 5)
    ax.set_ylim(min(gt_y) - 16, max(gt_y) + 8)

    cax = plt.axes([0.85, 0.1, 0.05, 0.7])
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label("z distance to ground truth [m]", fontsize=14)

    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.2)
    info_text = (
        r"$\mathbf{Horizontal\ Error\ (x-y\ plane):}$" + "\n"
        f"Mean x-y distance to ground truth: {mean_dist[0]:.3f} m +/- {std_dist[0]:.3f} m\n"
        r"$\mathbf{Vertical\ Error\ (z\ direction):}$" + "\n"
        f"Mean z distance to ground truth: {mean_dist[1]:.3f} m +/- {std_dist[1]:.3f} m\n"
        r"$\mathbf{3d\ Error :}$" + "\n"
        f"Mean 3d distance to ground truth: {mean_dist[2]:.3f} m +/- {std_dist[2]:.3f} m"
    )
    plt.figtext(0.15, 0.025, info_text, fontsize=14, bbox=dict(facecolor="white", alpha=0.7), transform=ax.transAxes)
    plt.savefig(f"{output_path}/m{model_index}_mean_std_{title}.png")
    plt.show()


def create_box_plot(distances, filenames, output_path, model_index):
    num_plots = len(filenames)

    fig, axes = plt.subplots(nrows=1, ncols=num_plots, tight_layout=True)
    if num_plots == 1:
        axes = [axes]

    labels = ["Batch \,Init \, Error", "Batch \,Aligned \, Error"]
    for i in range(1, len(filenames) - 1):
        labels.append("Model \," + str(i))

    cmap = colormaps.get_cmap("viridis")
    color_blue = cmap(0.5)
    color_outliers = cmap(0.1)
    color_median = cmap(0.9)
    for index, ax in enumerate(axes):
        sns.boxplot(
            distances[index],
            notch=False,
            showfliers=False,
            ax=ax,
            boxprops=dict(facecolor="white", alpha=1),
            medianprops=dict(color=color_median),
            flierprops=dict(marker="o", markersize=2, markerfacecolor=color_outliers, markeredgecolor=color_outliers),
        )
        sns.stripplot(distances[index], color=color_blue, size=2, jitter=True, ax=ax)
        ax.xaxis.set_ticks_position("none")
        ax.grid(color="grey", axis="y", linestyle="-", linewidth=0.25, alpha=0.5)

        percentiles = np.percentile(distances[index], [25, 50, 75])
        label_text = (
            r"$\bf{"
            + labels[index]
            + "}$\n"
            + r"$\bf{Median: }$"
            + f"{percentiles[1]:.3f} m\n"
            + r"$\bf{25 \% Percentile: }$"
            + f"{percentiles[0]:.3f} m\n"
            + r"$\bf{75 \% Percentile: }$"
            + f"{percentiles[2]:.3f} m\n"
        )
        ax.set_xlabel(label_text)

        # # Set the same scale for all plots
        # all_data = np.concatenate(distances)
        # ax.set_ylim(np.min(all_data), np.max(all_data))
        ax.set_ylim(0.05, 2)

    axes[0].set_ylabel("L2-Error between GT and Estimated Tree positions [m]")

    plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
    plt.savefig(f"{output_path}/m{model_index}_boxplot_batch.png")  # , dpi=300, transparent=True, bbox_inches='tight'
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth_positions", help="Path to csv script containing ground truth, header: x,y,z,tree_name"
    )
    parser.add_argument(
        "--tree_positions",
        nargs="*",
        help="Path to the csv file that stores the trees for the various models, "
        "should have header x,y,z,tree_name. Note that the third command line "
        "should contain all the models in one file the fourths should correspond "
        "to the base model",
    )
    parser.add_argument("--output_path", help="path to save the eval plot")
    parser.add_argument("--model_index", help="idx of current evaluated model")
    parser.add_argument("--tree_pair_file", help="Path of the file saving the tree pais info")
    parser.add_argument("--method_names", nargs="*", help="List of method names such as icp, FGR, etc.")
    args = parser.parse_args()

    OUTPUT_PATH = f"{args.output_path}"
    model_index = int(args.model_index)
    tree_pairs = pd.read_csv(f"{args.tree_pair_file}", header=0)
    ground_truth_trees = pd.read_csv(args.ground_truth_positions, delimiter=" ", header=0)

    all_distances = []
    filenames = []
    all_model_stats = []

    # method_names = ['colmap_init','initial aligned','loop aligned','batch aligned'] # our steps
    # method_names = ['icp', 'FGR', 'fastICP', 'robustICP','teaser++'] # SOTA
    method_names = args.method_names

    for title_i, tree_position in enumerate(args.tree_positions):
        print("Filename: ", tree_position)
        filenames.append(tree_position)
        estimated_trees = pd.read_csv(tree_position, delimiter=" ", header=0)

        # ------------------------------------------------------------------
        # -         average distance to ground truth tree position         -
        # ------------------------------------------------------------------

        distances = []
        sample_z_distance = np.zeros(len(estimated_trees))

        for row in tree_pairs.itertuples(index=False):
            i = getattr(row, "Index_base")  # Get the base index
            j = getattr(row, "Index_model")

            # get ground truth position for tree i
            gt_list = ground_truth_trees.loc[ground_truth_trees["tree_idx"] == i, ["x", "y", "z"]].values

            # get all estimated positions for that tree
            est_list = estimated_trees.loc[estimated_trees["tree_idx"] == j, ["x", "y", "z"]].values
            z_indices = estimated_trees.loc[estimated_trees["tree_idx"] == j, "z"].index.tolist()

            euclidean_x_y_dist = np.linalg.norm(est_list[:, :2] - gt_list[0, :2], axis=1)
            euclidean_x_y_z_dist = np.linalg.norm(est_list[:, :] - gt_list[0, :], axis=1)
            absolute_z_dist = np.abs(est_list[:, 2] - gt_list[0, 2])

            for index, z_error in enumerate(absolute_z_dist):
                sample_z_distance[z_indices[index]] = z_error

            dist = np.concatenate(
                [
                    np.expand_dims(euclidean_x_y_dist, axis=1),
                    np.expand_dims(absolute_z_dist, axis=1),
                    np.expand_dims(euclidean_x_y_z_dist, axis=1),
                ],
                axis=1,
            )

            distances.append(dist)

        # mean of distances
        dist_arr = np.concatenate(distances, axis=0)

        np.savetxt(
            f"{OUTPUT_PATH}/tree_error_{method_names[title_i]}_m{model_index}.txt",
            dist_arr,
            header="x-y,z,xyz",
            delimiter=",",
            comments="",
        )
        # total_dist_arr = np.concatenate(total_distances, axis=0)
        mean_dist = np.mean(dist_arr, axis=0)
        std_dist = np.std(dist_arr, axis=0)  # [:,:2]
        median_dist = np.median(dist_arr, axis=0)

        print("Errors for Model: ", tree_position)
        print(f"mean (x,y) distance is: {mean_dist[0]} with a standard deviation of +/- {std_dist[0]}")
        print(f"mean z distance is: {mean_dist[1]} with a standard deviation of +/- {std_dist[1]}")
        print(f"mean 3d distance is: {mean_dist[2]} with a standard deviation of +/- {std_dist[2]}")

        median_distances_content = {
            "idx": f"{method_names[title_i]}",
            "Median_XY_Dist": {median_dist[0]},
            "Median_Z_Dist": {median_dist[1]},
            "Median_XYZ_Dist": {median_dist[2]},
        }
        all_model_stats.append(median_distances_content)

        title = f"{method_names[title_i]}"
        plot_trees_in_x_y_plane(
            ground_truth_trees,
            estimated_trees,
            sample_z_distance,
            median_dist,
            mean_dist,
            std_dist,
            OUTPUT_PATH,
            title,
            model_index,
        )

        all_distances.append(dist_arr[:, 2])  # euclidean_x_y_z_dist

    create_box_plot(all_distances, filenames, OUTPUT_PATH, model_index)
    errors_df = pd.DataFrame(all_model_stats)
    errors_df.to_csv(OUTPUT_PATH + f"/errors_stats_m{model_index}.csv", index=False)


if __name__ == "__main__":
    main()
