# THE DRAGN GRAPH WRAPPER CLASS IMPLEMENTING THE NETWORKX PACKAGE

import networkx as nx
import matplotlib.pyplot as plt

# HELPFUL LINK: https://networkx.org/documentation/stable/tutorial.html

class DGraph():
    def __init__(self):
        self.G = nx.DiGraph()
        pass
    def nodes(self):
        return self.G.nodes()

    def edges(self):
        return self.G.edges()

    # ADDING TO THE GRAPH
    def addNode(self, input):
        self.G.add_node(input)
        pass

    def addEdge(self, tuple):
        self.G.add_edge(*tuple)
        pass

    def add_multiple_nodes(self, input_list):
        self.G.add_nodes_from(input_list)
        pass

    def add_multiple_edges(self, tuple_list):
        self.G.add_edges_from(tuple_list)
        pass
    
    def add_multiple_weighted_edges(self, triple_list):
        self.G.add_weighted_edges_from(triple_list)
        pass
    
    def removeNode(self):
        pass
    def editNode(self):
        pass
    def display(self):
        A = nx.nx_agraph.to_agraph(self.G)
        A.layout()
        A.draw('networkx_graph.png')
        pass

    def plot(self, with_labels_in=True, font_weight_in='bold'):
        nx.draw(self.G, with_labels=with_labels_in, font_weight=font_weight_in)
        plt.show()
        pass
    
    def plot_weighted(self, labels='weight', with_labels_in=True, font_weight_in='bold'):
        pos=nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=with_labels_in, font_weight=font_weight_in)
        edge_attr = nx.get_edge_attributes(self.G, labels)
        print(edge_attr)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_attr)
        plt.show()

    def to_json(self):
        pass

    def from_json(self):
        pass