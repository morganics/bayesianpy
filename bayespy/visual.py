import networkx as nx
from bayespy.jni import bayesServer

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