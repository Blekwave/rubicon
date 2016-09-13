import os.path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("white")
except ImportError as e:
    print("Seaborn is unavailable - default color scheme will be used")


def plot_records(path, records):
    fontsize = 'large'
    mpl.rc('axes', labelsize=fontsize)
    mpl.rc('xtick', labelsize=fontsize)
    mpl.rc('ytick', labelsize=fontsize)
    if isinstance(records[0], tuple):
        fields = records[0]._fields
        legend_handles = []
        gens = len(records)
        for values, field in zip(zip(*records), fields):
            handle, = plt.plot(range(gens), values, label=field)
            legend_handles.append(handle)
        legend = plt.legend(handles=legend_handles, loc='upper right',
                            frameon=True, fontsize=fontsize, framealpha=0.5)
        legend.get_frame().set_facecolor('#FFFFFF')
    else:
        plt.plot(range(len(records)), records)
    plt.xlabel('Generation')
    plt.savefig(path)
    plt.clf()


def plot_graphs(stats, run_dir, file=sys.stdout):
    for stat_name, records in stats.items():
        filename = "{}.pdf".format(stat_name)
        path = os.path.join(run_dir, filename)
        plot_records(path, records)
