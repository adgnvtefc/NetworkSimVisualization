import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Seaborn and matplotlib style
sns.set_style('white')
sns.set_context('paper', font_scale=1)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'axes.titlesize': 16,
    'axes.titlepad': 10,
    'axes.labelsize': 20,
    'axes.labelpad': 8,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'xtick.major.pad': 6,
    'ytick.major.pad': 6,
    'lines.linewidth': 1.0,
    'axes.grid': False,
    'figure.subplot.left': 0.15,
    'figure.subplot.right': 0.95,
    'figure.subplot.bottom': 0.20,
    'figure.subplot.top': 0.95,
    'text.usetex': False
})

# Formal names and styling maps
formal_name = {
    'graph': 'GNN',
    'dqn': 'DQN',
    'tabular': 'Tabular'
}
colors = {
    'graph':   '#1f77b4',  # blue
    'dqn':     '#ff7f0e',  # orange
    'tabular': '#17becf'   # cyan
}
markers = {
    'graph':   'o',  # circle
    'dqn':     's',  # square
    'tabular': '*'   # star
}
linestyles = {
    'graph':   '-',
    'dqn':     '-',
    'tabular': '-'
}

# Legend order
legend_order = ['graph', 'dqn', 'tabular']

def plot_compute_cost(
    nodes,
    tab_time,
    dqn_time,
    gnn_time,
    output_dir="real_data_trials/c_cost",
    file_name="computational_cost",
    textwidth_inches=7.0,
    auto_scale=True
):
    """
    Plot computational cost (seconds) vs. number of nodes:
      - Matches layout & styling of main plots
      - Fixed color, marker, and linestyle per algorithm
      - Removes grid, optional autoscale
      - Consistent font sizes and margins
      - Saves as PDF
    Args:
      auto_scale (bool): if False, disable y-axis autoscaling
    """
    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare figure
    width, height = textwidth_inches, textwidth_inches * 0.6
    fig, ax = plt.subplots(figsize=(width, height))

    # Helper to style axes
    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    # Plot lines in legend_order: GNN, DQN, Tabular
    finals = []
    series = {
        'graph': gnn_time,
        'dqn':   dqn_time,
        'tabular': tab_time
    }
    for key in legend_order:
        data = series[key]
        ax.plot(
            nodes, data,
            label=formal_name[key],
            color=colors[key],
            marker=markers[key],
            markersize=4,
            linestyle=linestyles[key]
        )
        finals.append(data[-1])

    # Optional autoscale Y-axis based on final values
    if auto_scale and finals:
        lo, hi = min(finals), max(finals)
        margin = 0.1 * (hi - lo) if hi > lo else lo * 0.1
        ax.set_ylim(lo - margin, hi + margin)

    # Labels and title
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Time (seconds)")

    # Style axes and legend
    style_ax(ax)
    ax.legend(frameon=False)

    ax.set_xticks(nodes)
    ax.set_xlim(nodes[0], nodes[-1])

    # Save as PDF
    out_path = os.path.join(output_dir, f"{file_name}.pdf")
    fig.savefig(out_path, format='pdf', dpi=300)
    plt.close(fig)

    print(f"Saved computational cost plot to {out_path}")

if __name__ == '__main__':
    nodes = list(range(2, 12))
    tab_time = [0.0071, 0.0088, 0.0354, 0.1049, 0.3490,
                0.9060, 2.9397, 6.5880, 17.9033, 44.8721]
    dqn_time = [9.7155, 5.7668, 6.5936, 8.6184, 9.6514,
                10.0265, 11.0897, 11.5243, 13.0774, 13.8299]
    gnn_time = [6.0337, 0.2538, 49.3024, 0.8811, 80.2261,
                33.8758, 72.8039, 67.1614, 47.6199, 45.6080]
    dqn_epoch = 3
    gnn_epoch = 20
    # average per epoch
    dqn_time = [t / dqn_epoch for t in dqn_time]
    gnn_time = [t / gnn_epoch for t in gnn_time]

    # call with auto_scale=True or False
    plot_compute_cost(nodes, tab_time, dqn_time, gnn_time, auto_scale=False)
