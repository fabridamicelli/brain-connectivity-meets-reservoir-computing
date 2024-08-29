from itertools import product
import os
from pprint import pprint

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

import bioRNN.plotting as plot
from bioRNN.utils import refactor_names


def set_context_and_font():
    sns.set_style("ticks")
    sns.set_context(
        "paper", 
        font_scale=3.,
    )
    plt.rcParams["axes.facecolor"] = ".97"
    plt.rcParams['font.family'] = 'serif'
    set_rcParams_color()

    
def set_rcParams_color(color=".3"):
    properties = (
        "text.color",
        "xtick.color",
        "ytick.color",
        "axes.titlecolor",
        "axes.labelcolor",
        "axes.edgecolor",
        "figure.edgecolor"         
    )
    for property_ in properties:
        plt.rcParams[property_] = color


my_palette = sns.color_palette("Spectral", 49,)# desat=.9)
my_palette = [
    my_palette[5],
    my_palette[12],
    my_palette[22],
    #my_palette[34],
    my_palette[39],
    my_palette[46],
]

# Define some custom colors
orange = my_palette[1]
grey = ".45"
green = my_palette[3]
yellow = my_palette[2]



def set_spines_width(axes, width=4):
    for i, ax in enumerate(axes.flat):
        ax.spines['left'].set_linewidth(width)
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)
        ax.spines['top'].set_linewidth(width)

def set_grid(
    axes,
    linestyle="--", 
    alpha=0.8, 
    linewidth=2.5,
    color="#b0b0b0",
    axis="both"
):
    for ax in axes.flat:
        ax.grid(
            True,
            linestyle=linestyle,
            alpha=alpha,
            color=color,
            linewidth=linewidth,
            axis=axis,
    )

def remove_ticks(axes):
    """Remove tick of x and y axis for all axes"""
    for ax in axes.flat:
        ax.yaxis.set_tick_params(left=False,)
        ax.xaxis.set_tick_params(bottom=False,)
    
def rotate_xlabels(axes, rotation=35):
    for ax in axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
        
def set_facetgrid_titles(
    g: sns.FacetGrid,
    col_template=r'Input scaling ($\epsilon$) = {col_name}',
    row_template=r"Leak rate ($\alpha$) = {row_name}", 
    size=25,
) -> None:
    """Clean first and set up columns and rows titles"""
    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")
    g.set_titles(col_template=col_template, row_template=row_template, size=size)
        
        
def tweak_axes(
    axes, 
    grid_axis="both", 
    grid_alpha=.8,
    grid_color="#b0b0b0",
):
    set_spines_width(axes)
    set_grid(axes,
             axis=grid_axis, 
             alpha=grid_alpha,
             color=grid_color,
            )
    remove_ticks(axes)
    sns.despine(right=False, top=False)


names_map = dict(zip(
    ['W_bio_rank', 'W_bio_norank', 'W_rnd_density', 'W_rnd_k', 'W_rnd_full'],
    ['Bio (rank)', 'Bio (no-rank)', 'Random (density)', 'Random (k)', 'Random (full)'],
))

def make_legend_kwargs(Ws, palette, alpha=1):
    """Create kwargs to generate a legend"""
    legend_kwargs = dict(
        title="W",
        handles=[Patch(color=c, label=names_map[W_name], alpha=alpha)
                 for W_name, c in zip(Ws.keys(), palette)],
        markerscale=2.5,
        handletextpad=.5,
        fontsize=30,
        title_fontsize=35,
     #   bbox_to_anchor=(.52, .1, .45, .75),
        facecolor="white",
        edgecolor="white",
    )
    
    return legend_kwargs