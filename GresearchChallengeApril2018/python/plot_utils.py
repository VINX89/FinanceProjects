#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm as cm

def plot_corr_mat(df):
    labels=list(df)
    labels.remove('Weight')
    labels.remove('y')

    corr_matrix = df[labels].corr()
    
    jump_x = corr_matrix.shape[0]*1.0/len(labels)
    jump_y = corr_matrix.shape[1]*1.0/len(labels)
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(corr_matrix, 
                     interpolation="nearest", 
                     cmap=cmap)
    ax1.set_xticklabels(labels,fontsize=16, rotation='vertical')
    ax1.set_yticklabels(labels,fontsize=16)
    ax1.set_xticks(np.arange(0,corr_matrix.shape[0], jump_x))
    ax1.set_yticks(np.arange(0,corr_matrix.shape[1], jump_y))
    ax1.tick_params('both', length=0, width=0, which='major')
    ax1.tick_params('both', length=0, width=0, which='minor')
    ax1.set_aspect('auto')
    fig.colorbar(cax)

    x_positions = np.linspace(start=0, stop=len(labels), num=len(labels), endpoint=False)
    y_positions = np.linspace(start=0, stop=len(labels), num=len(labels), endpoint=False)
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            x_feat = labels[x_index]
            y_feat = labels[y_index]
            c = "{0:.2f}".format(df[x_feat].corr(df[y_feat]))
            ax1.text(x, y, str(c), {'size': 12}, color='black', ha='center', va='center')

    return fig

def compare_weights(df):
    #1D projections (weighted VS unweighted)
    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(25,20))
    labels=list(df)
    labels.remove('Weight')
    idx=0
    for row in range(5):
        for col in range(3):
            label=labels[idx]
            ax = axs[row,col];
            _ = ax.hist(df[label], color='blue', alpha=0.3, normed=True, weights=None);
            _ = ax.hist(df[label], color='red', alpha=0.3, normed=True, weights=df.Weight.values);
            ax.set_xlabel(label);
            if "x" in label or "y" in label:
                ax.set_yscale('log');
            patch0 = mpatches.Patch(color='blue',
                                    label='Unweighted',
                                    alpha=0.3);
            patch1 = mpatches.Patch(color='red',
                                    label='Weighted',
                                    alpha=0.3);
            ax.legend(loc='best', handles=[patch0,patch1]);
            idx += 1
    plt.minorticks_on()
    plt.tight_layout()
    
    return fig

def marketwise_proj(df):
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(20,30))
    labels=list(df)
    labels.remove('Weight')
    labels.remove('Market')
    idx=0
    for row in range(7):
        for col in range(2):
            label=labels[idx]
            ax = axs[row,col];
            if "x" in label or "y" in label:
                ax.set_yscale('log');
                bins = 10
                _,bins,_ = ax.hist(df.query("Market==1")[label], bins=bins, color='blue', alpha=0.3, normed=True, weights=df.query("Market==1").Weight.values);
                _,bins,_ = ax.hist(df.query("Market==2")[label], bins=bins, color='red', alpha=0.3, normed=True, weights=df.query("Market==2").Weight.values);
                _,bins,_ = ax.hist(df.query("Market==3")[label], bins=bins, color='black', alpha=0.3, normed=True, weights=df.query("Market==3").Weight.values);
                _,bins,_ = ax.hist(df.query("Market==4")[label], bins=bins, color='green', alpha=0.3, normed=True, weights=df.query("Market==4").Weight.values);
            else:
                _ = ax.hist(df.query("Market==1")[label], color='blue', alpha=0.3, normed=True, weights=df.query("Market==1").Weight.values);
                _ = ax.hist(df.query("Market==2")[label], color='red', alpha=0.3, normed=True, weights=df.query("Market==2").Weight.values);
                _ = ax.hist(df.query("Market==3")[label], color='black', alpha=0.3, normed=True, weights=df.query("Market==3").Weight.values);
                _ = ax.hist(df.query("Market==4")[label], color='green', alpha=0.3, normed=True, weights=df.query("Market==4").Weight.values);
            ax.set_xlabel(label);
            patch1 = mpatches.Patch(color='blue',
                                    label='Market 1',
                                    alpha=0.3);
            patch2 = mpatches.Patch(color='red',
                                    label='Market 2',
                                    alpha=0.3);
            patch3 = mpatches.Patch(color='black',
                                    label='Market 3',
                                    alpha=0.3);
            patch4 = mpatches.Patch(color='green',
                                    label='Market 4',
                                    alpha=0.3);
            ax.legend(loc='best', handles=[patch1,patch2,patch3,patch4]);
            idx += 1
    plt.minorticks_on()
    plt.tight_layout()
    
    return fig

def scatter_plots(df):
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(20,30))
    labels=list(df)
    if 'Weight' in labels:
        labels.remove('Weight')
    labels.remove('y')
    idx=0
    for row in range(7):
        for col in range(2):
            label=labels[idx]
            ax = axs[row,col];
            ax.scatter(df[label],df["y"])
            ax.set_xlabel(label);
            ax.set_ylabel("y");
            c = "correlation = {0:.4f}".format(df[label].corr(df["y"]))
            ax.text(0.5, 0.85, str(c), {'size': 20}, transform=ax.transAxes, color='black', ha='right', va='bottom');
            idx += 1
    plt.minorticks_on()
    plt.tight_layout()
    
    return fig

def time_series(df):
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(20,30))
    labels=list(df)
    if 'Weight' in labels:
        labels.remove('Weight')
    labels.remove('Stock')
    labels.remove('Day')
    labels.remove('Market')
    idx=0
    for row in range(6):
        for col in range(2):
            label=labels[idx]
            ax = axs[row,col];
            ax.plot(df.query("Market==1")["Day"], df.query("Market==1")[label], color='blue')
            ax.plot(df.query("Market==2")["Day"], df.query("Market==2")[label], color='red')
            ax.plot(df.query("Market==3")["Day"], df.query("Market==3")[label], color='black')
            ax.plot(df.query("Market==4")["Day"], df.query("Market==4")[label], color='green')
            ax.set_xlabel("time");
            ax.set_ylabel(label);
            patch1 = mpatches.Patch(color='blue',
                                    label='Market 1',
                                    alpha=0.3);
            patch2 = mpatches.Patch(color='red',
                                    label='Market 2',
                                    alpha=0.3);
            patch3 = mpatches.Patch(color='black',
                                    label='Market 3',
                                    alpha=0.3);
            patch4 = mpatches.Patch(color='green',
                                    label='Market 4',
                                    alpha=0.3);
            ax.legend(loc='best', handles=[patch1,patch2,patch3,patch4]);
            idx += 1
    plt.minorticks_on()
    plt.tight_layout()
    
    return fig