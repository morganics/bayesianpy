import networkx as nx
import numpy as np
from bayesianpy.jni import bayesServer

import bayesianpy.data
import pandas as pd

import math
import scipy.stats as ss
from typing import List, Dict

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
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

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
                                          results: Dict[str, bayesianpy.model.Distribution]):

        hv = head_variables

        ax.plot(df[hv[0]].tolist(), df[hv[1]].tolist(), 'bo')
        #ax.set_title("{} vs {}".format(hv[0], hv[1]))
        for k, v in results.items():
            self._plot_cov_ellipse(cov=v.get_cov_by_variable(hv[0], hv[1]),
                                   pos=v.get_mean_by_variable(hv[0], hv[1]),
                                   nstd=3, alpha=0.5, color='green', ax=ax)

        ax.set_xlim([df[hv[0]].min() - 3, df[hv[0]].max() + 3])
        ax.set_ylim([df[hv[1]].min() - 3, df[hv[1]].max() + 3])
        ax.set_xlabel(hv[0])
        ax.set_ylabel(hv[1])

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