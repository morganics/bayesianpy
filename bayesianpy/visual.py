import networkx as nx
import numpy as np
from bayesianpy.jni import bayesServer

import bayesianpy.data
import pandas as pd

import math
import scipy.stats as ss
from typing import List, Dict
import sklearn.metrics

import logging

class NetworkLayout:
    def __init__(self, jnetwork):
        self._jnetwork = jnetwork
        self._graph = None
        self._multiplier = 500

    def build_graph(self):
        g = nx.DiGraph()
        for node in self._jnetwork.getNodes():
            g.add_node(node.getName())

        for link in self._jnetwork.getLinks():
            fr = link.getFrom().getName()
            to = link.getTo().getName()
            g.add_edge(fr, to)

        return g

    def visualise(self, graph, pos):
        import pylab
        nx.draw_networkx_nodes(graph, pos)
        nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='w')
        pylab.show()

    def spring_layout(self, graph):
        pos = nx.spring_layout(graph,center=[0.5,0.5])
        return pos

    def fruchterman_reingold_layout(self, graph):
        return nx.fruchterman_reingold_layout(graph,center=[0.5,0.5])

    def circular_layout(self, graph):
        return nx.circular_layout(graph, center=[0.5,0.5])

    def random_layout(self, graph):
        return nx.random_layout(graph,center=[0.5,0.5])

    def update_network_layout(self, pos):
        for key, value in pos.items():
            node = self._jnetwork.getNodes().get(key)
            b = node.getBounds()
            height = b.getHeight()
            width = b.getWidth()
            x = value[0]*self._multiplier
            y = value[1]*self._multiplier
            if x < 0:
                x = 0.0
            if y < 0:
                y = 0.0
            node.setBounds(bayesServer().Bounds(x, y, width, height))

class JointDistribution:

    # http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    @staticmethod
    def _plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the
        ellipse patch artist.

        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import Ellipse

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        if ax is None:
            ax = plt.gca()

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                        **kwargs)
        ax.add_artist(ellip)
        return ellip

    def plot_distribution_with_variance(self, ax, df: pd.DataFrame, head_variables: List[str],
                                        results: Dict[str, bayesianpy.model.Distribution]):
        import seaborn as sns
        for i, hv in enumerate(head_variables):
            x = np.arange(df[hv].min() - df[hv].std(), df[hv].max() + df[hv].std(), ((df[hv].max() + df[hv].std()) - (df[hv].min()-df[hv].std())) / 100)
            pdfs = [ss.norm.pdf(x, v.get_mean(), v.get_std()) for k, v in results.items()]
            density = np.sum(np.array(pdfs), axis=0)
            ax.plot(x, density, label='Joint PDF', linestyle='dashed')
            ax.set_ylabel("pdf")
            for k, v in results.items():
                s = df
                for tv, st in v.get_tail():
                    s = s[s[tv] == bayesianpy.data.DataFrame.cast2(s[tv].dtype, st)]

                sns.distplot(s[hv], hist=False, label=v.pretty_print_tail(), ax=ax)

    def plot_distribution_with_covariance(self, ax, df: pd.DataFrame, head_variables: tuple,
                                          results: Dict[str, bayesianpy.model.Distribution], labels=None):

        hv = head_variables

        ax.plot(df[hv[0]].tolist(), df[hv[1]].tolist(), 'o', markeredgecolor='#e2edff', markeredgewidth=1,marker='o',
                                        fillstyle='full', color='#84aae8')
        #ax.set_title("{} vs {}".format(hv[0], hv[1]))
        for k, v in results.items():
            self._plot_cov_ellipse(cov=v.get_cov_by_variable(hv[0], hv[1]),
                                   pos=v.get_mean_by_variable(hv[0], hv[1]),
                                   nstd=3, edgecolor='#ffb24f', lw=2, facecolor='none',
                                   ax=ax)

        ax.set_xlim([df[hv[0]].min() - 3, df[hv[0]].max() + 3])
        ax.set_ylim([df[hv[1]].min() - 3, df[hv[1]].max() + 3])

        if labels is not None:
            label0 = labels[0]
            label1 = labels[1]
        else:
            label0 = hv[0]
            label1 = hv[1]

        ax.set_xlabel(label0)
        ax.set_ylabel(label1)

    def plot_with_variance(self, df: pd.DataFrame,
                           head_variables: List[str],
                           results: List[Dict[str, bayesianpy.model.Distribution]],
                           plots_per_page=6):

        import matplotlib.pyplot as plt
        cols = 2 if len(head_variables) > 1 else 1
        rows = math.ceil(len(head_variables) / cols)

        for i, r in enumerate(results):
            if i == 0 or k == plots_per_page:
                k = 0
                if i > 0:
                    yield fig
                    plt.close()
                fig = plt.figure(figsize=(12, 12))
                k += 1


            ax = fig.add_subplot(rows, cols, i + 1)
            self.plot_distribution_with_variance(ax, df, head_variables, r)

        yield fig
        plt.close()

    def plot_with_covariance(self, df: pd.DataFrame,
                             head_variables: List[str],
                             results: Dict[str, bayesianpy.model.Distribution],
                             plots_per_page=6):

        import matplotlib.pyplot as plt

        n = len(head_variables) - 1
        cols = 2
        total = (n * (n + 1) / 2) / cols

        k = 0
        for i, hv in enumerate(head_variables):
            for j in range(i + 1, len(head_variables)):
                if i == 0 or k == plots_per_page:
                    k = 0
                    if i > 0:
                        yield fig
                        plt.close()

                    fig = plt.figure(figsize=(12, 12))
                    k += 1

                ax = fig.add_subplot(total / 2, 2, k)
                self.plot_distribution_with_covariance(ax, df,
                                        (head_variables[i], head_variables[j]), results)

        yield fig

from matplotlib import pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        real_values = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        real_values = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f} ({:0.2f})".format(cm[i, j], real_values[i,j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

def _split_df(df, actual_col, predicted_col):
    rows = []
    for group in np.array_split(df, 10):
        score = sklearn.metrics.accuracy_score(group[actual_col].tolist(),
                                               group[predicted_col].tolist(),
                                               normalize=False)

        rows.append({'NumCases': len(group), 'NumCorrectPredictions': score})

    return pd.DataFrame(rows)

def calc_cumulative_gains(df: pd.DataFrame, actual_col: str, predicted_col:str, probability_col:str):
    df.sort_values(by=probability_col, ascending=True, inplace=True)

    subset = df[df[predicted_col] == True]
    lift = _split_df(subset, actual_col, predicted_col)

    #Cumulative Gains Calculation
    lift['RunningCorrect'] = lift['NumCorrectPredictions'].cumsum()
    lift['PercentCorrect'] = lift.apply(
        lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x['RunningCorrect'], axis=1)
    lift['CumulativeCorrectBestCase'] = lift['NumCases'].cumsum()
    lift['PercentCorrectBestCase'] = lift['CumulativeCorrectBestCase'].apply(
        lambda x: 100 if (100 / lift['NumCorrectPredictions'].sum()) * x > 100 else (100 / lift[
            'NumCorrectPredictions'].sum()) * x)
    lift['AvgCase'] = lift['NumCorrectPredictions'].sum() / len(lift)
    lift['CumulativeAvgCase'] = lift['AvgCase'].cumsum()
    lift['PercentAvgCase'] = lift['CumulativeAvgCase'].apply(
        lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x)

    #Lift Chart
    lift['NormalisedPercentAvg'] = 1
    lift['NormalisedPercentWithModel'] = lift['PercentCorrect'] / lift['PercentAvgCase']

    return lift

def plot_binned_response_rate(lift: pd.DataFrame):
    import seaborn as sns
    plt.figure()
    sns.barplot(y=lift['NumCorrectPredictions'] / lift['NumCases'], x=lift.index.tolist(), color='salmon', saturation=0.5)
    plt.show()

def plot_cumulative_gains(lift: pd.DataFrame):
    fig, ax = plt.subplots()
    fig.canvas.draw()

    handles = []
    handles.append(ax.plot(lift['PercentCorrect'], 'r-', label='Percent Correct Predictions'))
    handles.append(ax.plot(lift['PercentCorrectBestCase'], 'g-', label='Best Case (for current model)'))
    handles.append(ax.plot(lift['PercentAvgCase'], 'b-', label='Average Case (for current model)'))
    ax.set_xlabel('Total Population (%)')
    ax.set_ylabel('Number of Respondents (%)')

    ax.set_xlim([0, 9])
    ax.set_ylim([10, 100])
    try:
        labels = [int((label+1)*10) for label in [float(item.get_text()) for item in ax.get_xticklabels() if len(item.get_text()) > 0]]
    except BaseException as e:
        print([item.get_text() for item in ax.get_xticklabels()])

    ax.set_xticklabels(labels)

    fig.legend(handles, labels=[h[0].get_label() for h in handles])
    fig.show()

def plot_lift_chart(lift: pd.DataFrame):
    plt.figure()
    plt.plot(lift['NormalisedPercentAvg'], 'r-', label='Normalised \'response rate\' with no model')
    plt.plot(lift['NormalisedPercentWithModel'], 'g-', label='Normalised \'response rate\' with using model')
    plt.legend()
    plt.show()