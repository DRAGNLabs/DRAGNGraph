import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def get_page_kg(page_id, files):
    statements, properties, items = files

    # Get all connections
    outward_connections = statements[statements['source_item_id'] == page_id]
    inward_connections = statements[statements['target_item_id'] == page_id]
    if len(outward_connections) > 5:
        outward_connections = outward_connections[:5]
    if len(inward_connections) > 5:
        inward_connections = inward_connections[:5]
    connections = pd.concat([outward_connections, inward_connections])

    # Get all nodes
    inward_node_ids = inward_connections.source_item_id.values.tolist()
    outward_node_ids = outward_connections.target_item_id.values.tolist()
    node_ids = set(inward_node_ids + outward_node_ids + [page_id])

    # Get relevant node names from "items" which is a file
    frames = []
    for id in node_ids:
        frame = items[items['item_id'] == id]
        frames.append(frame)
    node_ids_to_labels = pd.concat(frames)

    # Get all edges
    inward_edge_ids = inward_connections.edge_property_id.tolist()
    outward_edge_ids = outward_connections.edge_property_id.tolist()
    edge_ids = set(inward_edge_ids + outward_edge_ids)

    # Get relevant edge names from "properties" which is a file
    frames = []
    for id in edge_ids:
        frame = properties[properties['property_id'] == id]
        frames.append(frame)
    edge_ids_to_labels = pd.concat(frames)

    # Substitute node_id into nodes
    item_id_replacement = node_ids_to_labels.set_index('item_id')['en_label'].to_dict()
    connections['source_item_id'] = connections['source_item_id'].replace(item_id_replacement)
    connections['target_item_id'] = connections['target_item_id'].replace(item_id_replacement)

    # Substitute property_id into edges
    edge_id_replacement = edge_ids_to_labels.set_index('property_id')['en_label'].to_dict()
    connections['edge_property_id'] = connections['edge_property_id'].replace(edge_id_replacement)

    # Clean up rows by those that have names and relationships filled out
    # Those that do not have numbers instead of names for nodes and connections
    # Though iterrrows() is not good practice (it is slow), here connections will
    #    always be really short so it will run fast
    to_drop = []
    for index, row in connections.iterrows():
        if isinstance(row[0], int) or isinstance(row[1], int) or isinstance(row[2], int):
            to_drop.append(index)
    connections = connections.drop(to_drop)
    triples = connections.values.tolist()
    G = make_graph(triples)
    return G, item_id_replacement[page_id]


def make_graph(triples):
    G = nx.DiGraph()
    triples_graph = [(triple[0], triple[2], {'relation': triple[1]}) for triple in triples]
    G.add_edges_from(triples_graph)
    return G


def show_graph(G):
    edge_labels = nx.get_edge_attributes(G, "relation")
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, pos=pos, connectionstyle='arc3, rad = 0.1')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()  # pause before exiting


def load_csv():
    statements = pd.read_csv('wiki_data/statements.csv').dropna()
    properties = pd.read_csv('wiki_data/property.csv').dropna()
    items = pd.read_csv('wiki_data/item.csv').dropna()
    files = [statements, properties, items]
    return files


def get_subgraph(root_node, graph):
    # Grab all nodes that are 2 connections out
    connections_outwards = 2
    nodes = []
    shortest_paths = nx.shortest_path(graph.to_undirected(), root_node)
    for k, v in shortest_paths.items():
        if len(v) <= connections_outwards + 1:
            nodes.append(k)

    # Find all edges between those list of nodes from the main H graph
    I = graph.subgraph(nodes)
    return I


def add_to_graph(graph, page_id, files):
    G, page_name = get_page_kg(page_id, files)
    return nx.compose(G, graph), page_name


def get_node_names(graph):
    return graph.nodes


if __name__ == "__main__":

    # THIS IS AN EXAMPLE OF USAGE

    files = load_csv()
    # Here the graph is made with the "United States of America" page (id of 30)
    G, page_name = get_page_kg(30, files) # G is the graph and 30 is the page id
    H = get_subgraph(page_name, G)
    show_graph(H)
    # Here the "Universe" page (id of 1) is added
    G, page_name = add_to_graph(G, 1, files)
    H = get_subgraph(page_name, G)
    show_graph(H)