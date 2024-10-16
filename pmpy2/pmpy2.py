import itertools
import argparse
import pathlib
from xml.etree import ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate


## GRAPH FORMAT
# ----------------------------------------------------------
default_draw_node = dict(
    node_size=1000
    )
default_draw_edge = dict(
    edge_color='grey',
    width=3,
    arrowsize=20,
    min_source_margin=20,
    min_target_margin=20
    )

config_subplot = dict(
    figsize=(10,8),
    layout='constrained'
    )
config_draw_node = dict(
    **default_draw_node,
    node_color='#1f78b4'
    )
config_draw_critical_node = dict(
    **default_draw_node,
    node_color='red'
    )
config_draw_edge = dict(
    **default_draw_edge,
    edge_color='grey'
    )
config_draw_critical_edge = dict(
    **default_draw_edge,
    edge_color='red'
    )
config_draw_labels = dict(
    font_weight='bold',
    font_color='black', 
    font_size=14
    )
config_draw_annotations = dict(
    xytext=(20,-80),
    textcoords='offset points',
    bbox={'boxstyle': 'round', 'fc': 'lightgrey'},
    fontsize=7,
    arrowprops={'arrowstyle': 'wedge'}
    )

## INPUT PARSER
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--xml', help='specify the xml file to parse data from')
#parser.add_argument('--csv', help='specify the csv file to parse data from')
args = parser.parse_args()

filepath = pathlib.Path.cwd() / pathlib.Path(args.xml)

xmltree = ET.parse(filepath)
cpm_xml = xmltree.getroot()

## NETWORKX GRAPH
# ----------------------------------------------------------
act_network = nx.DiGraph(name='Activity Network')

for child in cpm_xml:
    if child.tag == 'node':
        act_network.add_node(int(child.attrib['id']),
                             subset=child.attrib['subset'])
    if child.tag == 'arrow':
        act_network.add_edge(int(child.attrib['from']),
                             int(child.attrib['to']),
                             name=child.attrib['id'],
                             cost=int(child.attrib['dur']))

## SOLVING NETWORK
# ----------------------------------------------------------
activities = act_network.edges.data('cost')

begins = [n for n, d in act_network.in_degree() if d == 0]
ends = [n for n, d in act_network.out_degree() if d == 0]

critical_path = nx.dag_longest_path(act_network, weight='cost')
max_duration = nx.dag_longest_path_length(act_network, weight='cost')

# for-loops in Python are painful: https://stackoverflow.com/a/3611987
for begin in begins:
    for end in ends:
        if begin == end:
            continue
        
        # forward pass to calculate earliest start and finish times
        # This should be extracted to a function
        paths = nx.all_simple_edge_paths(act_network, source=begin, target=end)
        for path in paths:
            last_ef = 0
            for from_, to in path:
                dur = act_network.edges[from_, to]['cost']
                es = last_ef
                ef = es + dur
                last_ef = ef
                attrs = {(from_, to): {'es': es, 'ef': ef}}
                nx.set_edge_attributes(act_network, attrs)

        # backward pass to calculate latest start and finish times
        # all_simple_edge_paths must be called again to restart the generator
        paths = nx.all_simple_edge_paths(act_network, source=begin, target=end)
        for path in paths:
            last_ls = max_duration
            for from_, to in reversed(path):
                dur = act_network.edges[from_, to]['cost']
                lf = last_ls
                ls = lf - dur
                last_ls = ls
                attrs = {(from_, to): {'ls': ls, 'lf': lf}}
                nx.set_edge_attributes(act_network, attrs)

        # a third pass to calculate actibity float times
        # all_simple_edge_paths must be called again to restart the generator
        paths = nx.all_simple_edge_paths(act_network, source=begin, target=end)
        for path in paths:
            for a1, a2 in itertools.pairwise(path):
                from_, to = a1
                es = act_network.edges[from_, to]['es']
                es_next = act_network.edges[a2[0], a2[1]]['es']
                ef = act_network.edges[from_, to]['ef']
                ls = act_network.edges[from_, to]['ls']
                tfloat = ls - es
                ffloat = es_next - ef
                attrs = {(from_, to): {'tfloat': tfloat, 'ffloat': ffloat}}
                nx.set_edge_attributes(act_network, attrs)

            last_act = path[-1]
            es = act_network.edges[last_act[0], last_act[1]]['es']
            ef = act_network.edges[last_act[0], last_act[1]]['ef']
            ls = act_network.edges[last_act[0], last_act[1]]['ls']
            tfloat = ls - es
            ffloat = max_duration - ef
            attrs = {last_act: {'tfloat': tfloat, 'ffloat': ffloat}}
            nx.set_edge_attributes(act_network, attrs)

#activities = nx.generate_edgelist(act_network)
#for line in activities:
#    print(line)

## DRAW OUTPUT
# ----------------------------------------------------------
noncritical_path = [
    n
    for n in act_network.nodes
    if n not in critical_path
    ]
critical_edges = [
    (a, b)
    for a, b in itertools.pairwise(critical_path)
    ]
noncritical_edges = [
    (a, b)
    for a, b in act_network.edges
    if (a, b) not in critical_edges
    ]

datalist = []
edge_labels = {}
# `edge_data` is a tuple of (node1: int, node2: int, attributes: dict)
for edge_data in act_network.edges.data():
    edge = edge_data[0], edge_data[1]
    attrs = edge_data[2]
    label = '\n'.join([f'{k}: {v}' for k, v in attrs.items()])
    edge_labels[edge] = label

    datalist.append(attrs)

print(tabulate(datalist, headers='keys', tablefmt='plain'))

#edge_labels = nx.get_edge_attributes(act_network, 'cost')
pos = nx.multipartite_layout(act_network, subset_key='subset')
fig, ax = plt.subplots(**config_subplot)

nx.draw_networkx_nodes(act_network, pos=pos, ax=ax, nodelist=noncritical_path,
                       **config_draw_node)

nx.draw_networkx_nodes(act_network, pos=pos, ax=ax, nodelist=critical_path,
                       **config_draw_critical_node)

nx.draw_networkx_edges(act_network, pos=pos, ax=ax, edgelist=noncritical_edges,
                       **config_draw_edge)

nx.draw_networkx_edges(act_network, pos=pos, ax=ax, edgelist=critical_edges,
                       **config_draw_critical_edge)

nx.draw_networkx_labels(act_network, pos=pos, ax=ax, **config_draw_labels)

for edge in act_network.edges:
    n1_pos, n2_pos = pos[edge[0]], pos[edge[1]]
    xy = np.mean([n1_pos, n2_pos], axis=0)
    label = edge_labels[edge]
    ax.annotate(label, xy=xy, **config_draw_annotations)

plt.show()
