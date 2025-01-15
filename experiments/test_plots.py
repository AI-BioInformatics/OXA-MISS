#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
import matplotlib
from typing import Union, Optional, Iterable
import pandas as pd
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
csv_path = "/work/H2020DeciderFicarra/D2_4/chemorefractory/MultimodalDecider/dataloader/dataset/tcga_stad.csv"
df = pd.read_csv(csv_path)
filtered_df = df.drop_duplicates(subset='case_id')

ids_brca_path = "/work/H2020DeciderFicarra/D2_4/chemorefractory/MultimodalDecider/dataloader/dataset/brca_ids.csv"
ids_brca = pd.read_csv(ids_brca_path)
ids = ids_brca["Unnamed: 0"]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_time = 365*12
size = 10
fake_log_dict = {
    "all_event_times": np.random.randint(0, max_time, size=size).tolist(),
    "all_censorships": np.random.randint(0, 2, size=size).tolist(),
    "all_risk_scores": np.random.rand(size).tolist()
}
fake_bins = np.linspace(0, max_time, 5).astype(int).tolist()

log_dict = {
    "all_event_times": filtered_df["survival_months"].tolist(),
    "all_censorships": filtered_df["censorship"].tolist(),
    "all_risk_scores": np.random.rand(len(filtered_df)).tolist()
}
bins = np.linspace(0, filtered_df["survival_months"].max(), 5).astype(int).tolist()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def count_alive(times, events, milestones):
    counts = []
    for milestone in milestones:
        alive_count = np.sum((times >= milestone) | ((times < milestone) & (events == 1)))
        counts.append(alive_count)
    return counts

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def KaplanMeier_plot(log_dict, bins):        
    # params = {'mathtext.default': 'regular' }          
    # plt.rcParams.update(params)
    # font = {'family' : 'serif',
    #         'weight' : 'normal',
    #         'size'   : 8}
    # matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 6))


    all_event_times = np.array(log_dict["all_event_times"])
    all_censorships = np.array(log_dict["all_censorships"])
    all_risk_scores = np.array(log_dict["all_risk_scores"])
    mean_risk_score = np.mean(all_risk_scores)
    high_risk_events = all_event_times[all_risk_scores >= mean_risk_score]
    low_risk_events = all_event_times[all_risk_scores < mean_risk_score] 
    count_high = count_alive(high_risk_events, all_censorships[all_risk_scores >= mean_risk_score], bins)
    count_low = count_alive(low_risk_events, all_censorships[all_risk_scores < mean_risk_score], bins)

    print("count_high: ", count_high)
    print("count_low: ", count_low)


    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()

    kmf.fit(high_risk_events, event_observed=(1 - all_censorships[all_risk_scores >= mean_risk_score]))
    kmf.plot(ax=ax, label='High Risk', ci_force_lines = False, ci_show = False)
    # add_at_risk_counts(kmf, ax=ax, ypos=4.5, xticks=bins)
    
    kmf.fit(low_risk_events, event_observed=(1 - all_censorships[all_risk_scores < mean_risk_score]))
    kmf.plot(ax=ax, label='Low Risk', ci_force_lines = False, ci_show = False)

    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    ax.set_xticks(bins)

    # ax.set_xticklabels(bins)

    plt.title('Kaplan-Meier Survival Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    return plt
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = KaplanMeier_plot(log_dict, bins)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def move_spines(ax, sides, dists):
    """
    Move the entire spine relative to the figure.

    Parameters:
      ax: axes to operate on
      sides: list of sides to move. Sides: top, left, bottom, right
      dists: list of float distances to move. Should match sides in length.

    Example:
    move_spines(ax, sides=['left', 'bottom'], dists=[-0.02, 0.1])
    """
    for side, dist in zip(sides, dists):
        ax.spines[side].set_position(("axes", dist))
    return ax

def add_at_risk_counts(
    *fitters,
    labels: Optional[Union[Iterable, bool]] = None,
    rows_to_show=None,
    ypos=-0.6,
    xticks=None,
    ax=None,
    at_risk_count_from_start_of_period=False,
    **kwargs
):
    """
    Add counts showing how many individuals were at risk, censored, and observed, at each time point in
    survival/hazard plots.

    Tip: you probably want to call ``plt.tight_layout()`` afterwards.

    Parameters
    ----------
    fitters:
      One or several fitters, for example KaplanMeierFitter, WeibullFitter,
      NelsonAalenFitter, etc...
    labels:
        provide labels for the fitters, default is to use the provided fitter label. Set to
        False for no labels.
    rows_to_show: list
        a sub-list of ['At risk', 'Censored', 'Events']. Default to show all.
    ypos:
        make more positive to move the table up.
    xticks: list
        specify the time periods (as a list) you want to evaluate the counts at.
    at_risk_count_from_start_of_period: bool, default False.
        By default, we use the at-risk count from the end of the period. This is what other packages, and KMunicate suggests, but
        the same issue keeps coming up with users. #1383, #1316 and discussion #1229. This makes the adjustment.
    ax:
        a matplotlib axes

    Returns
    --------
      ax:
        The axes which was used.

    Examples
    --------
    .. code:: python

        # First train some fitters and plot them
        fig = plt.figure()
        ax = plt.subplot(111)

        f1 = KaplanMeierFitter()
        f1.fit(data)
        f1.plot(ax=ax)

        f2 = KaplanMeierFitter()
        f2.fit(data)
        f2.plot(ax=ax)

        # These calls below are equivalent
        add_at_risk_counts(f1, f2)
        add_at_risk_counts(f1, f2, ax=ax, fig=fig)
        plt.tight_layout()

        # This overrides the labels
        add_at_risk_counts(f1, f2, labels=['fitter one', 'fitter two'])
        plt.tight_layout()

        # This hides the labels
        add_at_risk_counts(f1, f2, labels=False)
        plt.tight_layout()

        # Only show at-risk:
        add_at_risk_counts(f1, f2, rows_to_show=['At risk'])
        plt.tight_layout()

    References
    -----------
     Morris TP, Jarvis CI, Cragg W, et al. Proposals on Kaplan–Meier plots in medical research and a survey of stakeholder views: KMunicate. BMJ Open 2019;9:e030215. doi:10.1136/bmjopen-2019-030215

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.gcf()
    if labels is None:
        labels = [f._label for f in fitters]
    elif labels is False:
        labels = [None] * len(fitters)
    if rows_to_show is None:
        rows_to_show = ["At risk", "Censored", "Events"]
    else:
        assert all(
            row in ["At risk", "Censored", "Events"] for row in rows_to_show
        ), 'must be one of ["At risk", "Censored", "Events"]'
    n_rows = len(rows_to_show)

    # Create another axes where we can put size ticks
    ax2 = plt.twiny(ax=ax)
    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    ax_height = (
        ax.get_position().y1 - ax.get_position().y0
    ) * fig.get_figheight()  # axis height
    ax2_ypos = ypos / ax_height

    move_spines(ax2, ["bottom"], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ["top", "right", "bottom", "left"])
    # Set ticks and labels on bottom
    ax2.xaxis.tick_bottom()
    # Set limit
    min_time, max_time = ax.get_xlim()
    ax2.set_xlim(min_time, max_time)
    # Set ticks to kwarg or visible ticks
    if xticks is None:
        xticks = [xtick for xtick in ax.get_xticks() if min_time <= xtick <= max_time]
    ax2.set_xticks(xticks)
    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)

    ticklabels = []

    for tick in ax2.get_xticks():
        lbl = ""

        # Get counts at tick
        counts = []
        for f in fitters:
            # this is a messy:
            # a) to align with R (and intuition), we do a subtraction off the at_risk column
            # b) we group by the tick intervals
            # c) we want to start at 0, so we give it it's own interval
            if at_risk_count_from_start_of_period:
                event_table_slice = f.event_table.assign(at_risk=lambda x: x.at_risk)
            else:
                event_table_slice = f.event_table.assign(
                    at_risk=lambda x: x.at_risk - x.removed
                )
            if not event_table_slice.loc[:tick].empty:
                event_table_slice = (
                    event_table_slice.loc[:tick, ["at_risk", "censored", "observed"]]
                    .agg(
                        {
                            "at_risk": lambda x: x.tail(1).values,
                            "censored": "sum",
                            "observed": "sum",
                        }
                    )  # see #1385
                    .rename(
                        {
                            "at_risk": "At risk",
                            "censored": "Censored",
                            "observed": "Events",
                        }
                    )
                    .fillna(0)
                )
                counts.extend([int(c) for c in event_table_slice.loc[rows_to_show]])
            else:
                counts.extend([0 for _ in range(n_rows)])
        if n_rows > 1:
            if tick == ax2.get_xticks()[0]:
                max_length = len(str(max(counts)))
                for i, c in enumerate(counts):
                    if i % n_rows == 0:
                        if is_latex_enabled():
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"\textbf{%s}" % labels[int(i / n_rows)]
                                + "\n"
                            )
                        else:
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"%s" % labels[int(i / n_rows)]
                                + "\n"
                            )
                    l = rows_to_show[i % n_rows]
                    s = (
                        "{}".format(l.rjust(10, " "))
                        + (" " * (max_length - len(str(c)) + 3))
                        + "{{:>{}d}}\n".format(max_length)
                    )

                    lbl += s.format(c)
            else:
                # Create tick label
                lbl += ""
                for i, c in enumerate(counts):
                    if i % n_rows == 0 and i > 0:
                        lbl += "\n\n"
                    s = "\n{}"
                    lbl += s.format(c)
        else:
            # if only one row to show, show in "condensed" version
            if tick == ax2.get_xticks()[0]:
                max_length = len(str(max(counts)))

                lbl += rows_to_show[0] + "\n"

                for i, c in enumerate(counts):
                    s = (
                        "{}".format(labels[i].rjust(10, " "))
                        + (" " * (max_length - len(str(c)) + 3))
                        + "{{:>{}d}}\n".format(max_length)
                    )
                    lbl += s.format(c)
            else:
                # Create tick label
                lbl += ""
                for i, c in enumerate(counts):
                    s = "\n{}"
                    lbl += s.format(c)
        ticklabels.append(lbl)
    # Align labels to the right so numbers can be compared easily
    ax2.set_xticklabels(ticklabels, ha="right", **kwargs)

    return ax
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
