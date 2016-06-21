import networkx as nx
from bayespy.jni import bayesServer
import bayespy.network
from matplotlib import pyplot as plt
import seaborn as sns
import bayespy.data
import math
import matplotlib.mlab as mlab
import numpy as np
import uuid
import os

class NetworkLayout:
    def __init__(self, jnetwork):
        self._jnetwork = jnetwork
        self._graph = None
        self._multiplier = 500

    def build_graph(self):
        g = nx.Graph()
        for node in self._jnetwork.getNodes():
            g.add_node(node.getName())

        for link in self._jnetwork.getLinks():
            fr = link.getFrom().getName()
            to = link.getTo().getName()
            g.add_edge(fr, to)

        return g

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
            node.setBounds(bayesServer.Bounds(x, y, width, height))

class VisualiseNetworkQuery:

    def __init__(self, data, insight, save_dir=None):
        self._data = data
        self._insight = insight
        self._save_path = save_dir

    def plot(self, model=None, variable_names=[], target=None):
        if model is None:
            raise ValueError("Need a model variable")

        folder = str(uuid.uuid4())

        if self._save_path is not None and not os.path.exists(os.path.join(self._save_path, folder)):
            os.makedirs(os.path.join(self._save_path, folder))

        for column in variable_names:
            
            node = bayespy.network.Discrete.fromstring(column)
            if node.variable == "Cluster" or node.variable == target.variable:
                continue
            fig1 = plt.figure()
            if bayespy.network.is_cluster_variable(column):

                ax = fig1.add_subplot(211)
                
                v_name = node.variable.replace("Cluster_", "")
                min_x = self._data[v_name].min()
                max_x = self._data[v_name].max()
                step = int((max_x - min_x) / 150)
                if step == 0:
                    step = 2

                for i in range(bayespy.network.get_number_of_states(model.get_network(), node.variable)):
                    color = "b"
                    if "Cluster{}".format(i) in column:
                        color = "r"

                    state = "Cluster{}".format(i)
                    e = bayespy.network.Discrete(node.variable, state)

                    (d, c) = self._insight.evidence_query(model=model, new_evidence=[e.tostring()])
                    mu = c[c.variable == e.variable.replace("Cluster_", "")].iloc[0]['mean']
                    sigma = math.sqrt(c[c.variable == e.variable.replace("Cluster_", "")].iloc[0]['variance'])
                    x = np.linspace(min_x, max_x, num=150)
                    ax.plot(x, mlab.normpdf(x, mu, sigma), color=color)
                    ax.set_xlim([min_x - step, max_x + step])

                ax.set_title(v_name)
                ax = fig1.add_subplot(212)

                try:
                    sns.distplot(self._data[self._data[target.variable] == target.state][v_name].dropna(), ax=ax, bins=range(int(min_x), int(max_x), step))
                    for other_state in bayespy.network.get_other_states_from_variable(model.get_network(), target):
                        other_state = bayespy.data.DataFrame.cast(self._data, v_name, bayespy.network.Discrete.fromstring(other_state).state)
                        sns.distplot(self._data[self._data[target.variable] == other_state][v_name].dropna(), ax=ax, bins=range(int(min_x), int(max_x), step))

                    ax.set_xlim([min_x - step, max_x + step])
                    if self._save_path is not None:
                        plt.savefig(os.path.join(self._save_path, folder, "{}.png".format(v_name)))
                except:
                    print(min_x, max_x, v_name)
            else:
                ax = fig1.add_subplot(111)
                sns.countplot(x=node.variable, hue=target.variable, data=self._data, ax=ax)
                ax.set_title("{}.{}".format(node.variable, node.state))
                # (fig, d) = mosaic(filtered, [n.variable, 'vds'], title="{}.{}".format(n.variable, n.state))
                if self._save_path is not None:
                    plt.savefig(os.path.join(self._save_path, folder, "{}${}.png".format(node.variable, node.state)))

        return os.path.join(self._save_path, folder)

