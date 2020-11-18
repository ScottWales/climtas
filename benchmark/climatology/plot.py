#!/g/data/hh5/public/apps/nci_scripts/python-analysis3
# Copyright 2020 Scott Wales
# author: Scott Wales <scott.wales@unimelb.edu.au>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas
import matplotlib.pyplot as plt


def plot_run(df, group, label, grid, ax, color, ls):
    df_group = df[df["name"] == group]

    df_group["metric"] = df_group["total"] ** 2 * df_group["client_workers"]
    df_group["metric"] = df_group["total"]

    mean = df_group.groupby("client_workers").mean()
    std = df_group.groupby("client_workers").std()

    mean["metric"].plot(label=label, ax=ax, color=color, ls=ls, yerr=std["metric"])


if __name__ == "__main__":

    prof_x = pandas.read_csv("climatology_xarray.csv")
    prof_c = pandas.read_csv("climatology_climtas.csv")

    prof_x["client_workers"] -= 0.1
    prof_c["client_workers"] += 0.1

    ax = plt.axes()

    plot_run(
        prof_x,
        "5 year 100x100 horiz",
        "xarray",
        100 ** 2,
        ax=ax,
        color="tab:blue",
        ls="-",
    )
    plot_run(
        prof_c,
        "5 year 100x100 horiz",
        "climtas",
        100 ** 2,
        ax=ax,
        color="tab:orange",
        ls="-",
    )

    ax.legend()

    plt.title(
        "Era5 Subset Mean Climatology Runtime\n5 year 100x100 gridpoints run at NCI Gadi"
    )
    plt.xlabel("Workers")
    plt.ylabel("Wall Time (s)\nLower is better")

    plt.ylim((0, plt.ylim()[1]))

    plt.savefig("climatology_walltime.png")

    plt.show()
