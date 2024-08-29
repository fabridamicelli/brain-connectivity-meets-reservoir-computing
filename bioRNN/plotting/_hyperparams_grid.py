import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def hyperparams_grid(
    *,
    data: pd.DataFrame,
    x: str = "spectral_radius",
    y: str = "memory_capacity",
    hue: str = "W",
    col: str = "input_scaling",
    row: str = "leak_rate",
    kind="box",
    margin_titles=True,
    legend=True,
    ci="sd",
    palette="Spectral",
    **catplot_kws,
):
    g = sns.catplot(
        x=x,
        y=y,
        hue=hue,
        col=col,
        row=row,
        kind=kind,
        margin_titles=margin_titles,
        data=data,
        palette=palette,
        legend=legend,
        ci=ci,
        **catplot_kws,
    )
    plt.setp(g._legend.get_title(), fontsize=30)
    for ax in g.axes.flat:
        ax.spines["left"].set_linewidth(4)
        ax.spines["bottom"].set_linewidth(4)
        ax.grid(linestyle="--", alpha=0.8, axis="y", linewidth=2.5)
        ax.yaxis.set_tick_params(width=3.5, size=10)
        ax.xaxis.set_tick_params(bottom=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35)
    sns.despine(
        trim=True,
        bottom=True,
    )
    return g
