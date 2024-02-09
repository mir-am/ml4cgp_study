'''
This code is borrowed from the following paper's repository with some modifications and speed improvements.
Striking a Balance: Pruning False-Positives from Static Call Graphs
'''

import sys
import queue
import pathlib
import copy
import json
import csv
import statistics
import random
import networkit as nk
import pandas as pd
from tqdm import tqdm
from typing import Dict
from collections import deque
from src.constants import struct_feat_names
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import MinMaxScaler

#Defining some constants
BENCHMARKS_FOLDER = "" #sys.argv[1]
'''This option reduces the dataset to an edge-level dataset 
(no call-site recorded. Only edges)
Across repeated edges (same src,dest but different offset), 
everything except fanout remains the same. Hence once we handle fanout,
we can simply throw out the repeated edges.
For fanout in this case now gets replaced by 'average_fanout' and 'min_fanout'
'min_fanout', which represents the fanout averaged (or min) across 
the repeated edges.
'''
REMOVE_OFFSETS = False #sys.argv[2]=="True"
KEEP_PROGRAM_LEVEL_COLS = True
# Removing this feature since it doesn't give good results
# -> not maintained AVERAGE_EXTRA_FEATURES = sys.argv[2] == "True"
DATASET_FILE = "combined_dataset.csv"
BENCHMARK_INFO_FILE = "benchmark.json"
DYNAMIC_ANALYSIS_NAME = "wiretap"
OUTPUT_DATASET_FILE = "combinationWithExtraFeatures.csv"
#We calculate the relative number of nodes/edges in the
#graph with this analysis' counts in the denominator
REFERENCE_ANALYSIS = "wala-cge-0cfa-noreflect-intf-direct"
UNREACHABLE = -1
#Beyond this cutoff for the number of edges in the graph, 
#some features will be skipped
LARGE_GRAPH_CUTOFF = 30000 
#Beyond this cutoff for the orphans in the graph, 
#some orhpan-related features will be skipped
LARGE_ORPH_COUNT_CUTOFF = 50 

'''Represents an edge of the call graph'''
class Edge:
    def __init__(self, a: str, b: str):
        self.bytecode_offset = a
        self.dest = b
        self.depth_from_main = -1
        self.depth_from_orphans = -1
        self.src_node_in_deg = 0 # Paper
        self.dest_node_out_deg = 0 # Paper
        self.dest_node_in_deg = 0 # Paper
        #src_node_out_deg is trivial
        # Fanout: no. of edges from a given source node,                    
        #with the same bytecode offset as this edge.
        self.fanout = 0  # Paper
        self.avg_fanout = 0 # Paper
        self.min_fanout = 0
        self.reachable_from_main = False
        self.num_paths_to_this_from_main = 0
        self.num_paths_to_this_from_orphans = 0
        self.repeated_edges = 0 # Paper
        self.node_disjoint_paths_from_main = 0
        self.node_disjoint_paths_from_orphans = 0
        self.edge_disjoint_paths_from_main = 0
        #-> calculating this next part takes too long
        #self.edge_disjoint_paths_from_orphans = -1

'''Represents a method of the call graph'''
class Node:
    def __init__(self):
        self.edges = set()
        self.depth = -1
        self.visited = False #temporary variable.

'''Represents a call graph of a static analysis'''
class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edge_count = 0 # Paper
        self.node_count = 0 # Paper
        self.relative_node_count = 0
        self.relative_edge_count = 0
        main_methodNode = ""
        self.avg_deg = 0 # Paper
        self.avg_edge_fanout = 0
        self.num_orphan_nodes = 0

'''Represents an edge in the union graph'''
class UnionEdge:
    def __init__(self, a: str, b: str, c: str, wiretap: str, wala: str):
        self.src = a
        self.bytecode_offset = b
        self.dest = c
        self.wiretap = wiretap
        self.wala = wala


def write_output(fp,csv_reader,union_edge_set,callgraphs):
    #Get the new columns and write out the whole header line
    full_header_line = (csv_reader.fieldnames +
                        get_new_column_headers(csv_reader.fieldnames))
    writer = csv.DictWriter(fp, fieldnames=full_header_line)
    writer.writeheader()

    for union_edge in union_edge_set:
        row = {} #the row to be written out  
        #First add the old columns that were read, as is
        add_old_entries_to_row(row,union_edge,callgraphs)
        #Now add the new columns
        #if AVERAGE_EXTRA_FEATURES:
        #    compute_output_averaged(row,union_edge,callgraphs)
        #else:    
        compute_output(row,union_edge,callgraphs)
        #Finally, write out the row to the file
        writer.writerow(row)


def add_old_entries_to_row(row,union_edge,callgraphs):
    #First add out the src, bytecode and dest for the edge
    row['method'] = union_edge.src
    if REMOVE_OFFSETS:
        row['offset'] = "xxx"
    else:
        row['offset'] = union_edge.bytecode_offset
    row['target'] = union_edge.dest

    #Next add, for each analysis, the old bit information of whether 
    #the edge exists according to the call graph
    for analysis_name,graph in callgraphs.items():
        #Check if the edge is present in the graph
        found_edge = False
        if union_edge.src in graph.nodes: 
            for edge2 in graph.nodes[union_edge.src].edges:
                if (edge2.dest==union_edge.dest
                        and edge2.bytecode_offset==union_edge.bytecode_offset):
                    found_edge = True
                    break
            
        #Print 0 or 1 depending on if the edge is there in the graph
        if found_edge:
            row[analysis_name] = "1"
        else:
            row[analysis_name] = "0"

def add_old_entries_to_row_imp(row,union_edge: UnionEdge):
    #First add out the src, bytecode and dest for the edge
    row['wiretap'] = union_edge.wiretap
    row['wala-cge-0cfa-noreflect-intf-trans'] = union_edge.wala
    row['method'] = union_edge.src
    row['offset'] = union_edge.bytecode_offset
    row['target'] = union_edge.dest


def get_new_column_headers(old_columns):
    '''Gets a list of the names of the new columns'''
    analysis_names = old_columns[3:]
    new_headers = []
    '''
    if AVERAGE_EXTRA_FEATURES:
        #There will be a common column for these 5 features.
        new_headers += ["edge_depth",
                        "src_node_in_deg",
                        "dest_node_out_deg",
                        "fanout",
                        "src_node_out_deg",
                        "reachable_from_main",
                        "num_paths_to_this",
                        
        ]
        #These 2 features will have a separate column for each analysis
        for analysis_name in analysis_names:
            #Don't add columns for the dynamic analysis
            if (DYNAMIC_ANALYSIS_NAME==analysis_name):
                continue
            new_headers += [analysis_name + "#graph_rel_node_count",
                        analysis_name + "#graph_rel_edge_count",
            ]
    '''

    #There will be separate column per feature per analysis
    for analysis_name in analysis_names:
        #Don't add columns for the dynamic analysis
        if (DYNAMIC_ANALYSIS_NAME==analysis_name):
            continue
        new_headers += [analysis_name + "#depth_from_main",
                    #analysis_name + "#depth_from_orphans",
                    analysis_name + "#src_node_in_deg",
                    analysis_name + "#dest_node_out_deg",
                    analysis_name + "#dest_node_in_deg",
                    analysis_name + "#src_node_out_deg",
                    #analysis_name + "#reachable_from_main", - redundant
                    #analysis_name + "#num_paths_to_this_from_main",
                    #analysis_name + "#num_paths_to_this_from_orphans",
                    analysis_name + "#repeated_edges",
                    analysis_name + "#node_disjoint_paths_from_main",
                    #analysis_name + "#node_disjoint_paths_from_orphans",
                    analysis_name + "#edge_disjoint_paths_from_main"#,
                    #analysis_name + "#edge_disjoint_paths_from_orphans", - too much time                 
        ]
        if REMOVE_OFFSETS:
            new_headers += [analysis_name + "#avg_fanout",
                            analysis_name + "#min_fanout"]
        else:
            new_headers += [analysis_name + "#fanout"] 

        if KEEP_PROGRAM_LEVEL_COLS:
            new_headers += [
                    #analysis_name + "#graph_rel_node_count",
                    #analysis_name + "#graph_rel_edge_count", 
                    analysis_name + "#graph_node_count",
                    analysis_name + "#graph_edge_count",
                    analysis_name + "#graph_avg_deg",
                    analysis_name + "#graph_avg_edge_fanout",
                    #analysis_name + "#graph_num_orphan_nodes"
                ]

    return new_headers


def compute_output(row, union_edge, callgraphs):
    '''Write out the new computed information on edge depths, etc.
    '''
    for analysis_name,graph in callgraphs.items():
        #Don't compute anything for the dynamic analysis
        if (DYNAMIC_ANALYSIS_NAME==analysis_name):
            continue
        #Check if the union_edge is present in the graph
        edge_in_graph = None
        if union_edge.src in graph.nodes: 
            for edge2 in graph.nodes[union_edge.src].edges:
                if (edge2.dest==union_edge.dest
                        and edge2.bytecode_offset==union_edge.bytecode_offset):
                    edge_in_graph = edge2
                    break
        
        #If the union_edge exists, write attribute values calculated
        if edge_in_graph is not None:
            if edge_in_graph.depth_from_main == UNREACHABLE:
                row[analysis_name + "#depth_from_main"] = "1000000000"
            else:
                row[analysis_name + "#depth_from_main"] = edge_in_graph.depth_from_main
            #if edge_in_graph.depth_from_orphans == UNREACHABLE:
            #    row[analysis_name + "#depth_from_orphans"] = "1000000000"
            #else:
            #    row[analysis_name + "#depth_from_orphans"] = edge_in_graph.depth_from_orphans
            row[analysis_name + "#src_node_in_deg"] = edge_in_graph.src_node_in_deg
            row[analysis_name + "#dest_node_out_deg"] = len(graph.nodes[union_edge.dest].edges)
            row[analysis_name + "#dest_node_in_deg"] = edge_in_graph.dest_node_in_deg
            row[analysis_name + "#src_node_out_deg"] = len(graph.nodes[union_edge.src].edges)
            #if edge_in_graph.reachable_from_main: - redundant
            #    row[analysis_name + "#reachable_from_main"] = 1
            #else:
            #    row[analysis_name + "#reachable_from_main"] = 0
            #row[analysis_name + "#num_paths_to_this_from_main"] = (
            #    float(edge_in_graph.num_paths_to_this_from_main))
            #row[analysis_name + "#num_paths_to_this_from_orphans"] = (
            #    float(edge_in_graph.num_paths_to_this_from_orphans))
            row[analysis_name + "#repeated_edges"] = edge_in_graph.repeated_edges
            row[analysis_name + "#node_disjoint_paths_from_main"] = (
                edge_in_graph.node_disjoint_paths_from_main)
            #row[analysis_name + "#node_disjoint_paths_from_orphans"] = (
            #    edge_in_graph.node_disjoint_paths_from_orphans)
            row[analysis_name + "#edge_disjoint_paths_from_main"] = (
                edge_in_graph.edge_disjoint_paths_from_main)
            #row[analysis_name + "#edge_disjoint_paths_from_orphans"] = (
                #edge_in_graph.edge_disjoint_paths_from_orphans)
            if REMOVE_OFFSETS:
                row[analysis_name + "#avg_fanout"] = edge_in_graph.avg_fanout
                row[analysis_name + "#min_fanout"] = edge_in_graph.min_fanout
            else:
                row[analysis_name + "#fanout"] = edge_in_graph.fanout
 
        #Else write out the default value - 
        #(because the final table cannot have empty cells)
        else: 
            row[analysis_name + "#depth_from_main"] = "1000000000"
            #row[analysis_name + "#depth_from_orphans"] = "1000000000"
            row[analysis_name + "#src_node_in_deg"] = "-1"
            row[analysis_name + "#dest_node_out_deg"] = "-1"
            row[analysis_name + "#dest_node_in_deg"] = "-1"
            row[analysis_name + "#src_node_out_deg"] = (
                "-1")
            #row[analysis_name + "#reachable_from_main"] = (
            #    "REACHABLE_FROM_MAIN_DEFAULT") -- redundant
            #row[analysis_name + "#num_paths_to_this_from_main"] = (
            #    "NUM_PATHS_TO_THIS_DEFAULT")
            #row[analysis_name + "#num_paths_to_this_from_orphans"] = (
            #    "NUM_PATHS_TO_THIS_DEFAULT")
            row[analysis_name + "#repeated_edges"] = "0"
            row[analysis_name + "#node_disjoint_paths_from_main"] = (
                "-1")
            #row[analysis_name + "#node_disjoint_paths_from_orphans"] = (
            #    "-1")
            row[analysis_name + "#edge_disjoint_paths_from_main"] = (
                "-1")
            #row[analysis_name + "#edge_disjoint_paths_from_orphans"] = (
                #"-1")
            if REMOVE_OFFSETS:
                row[analysis_name + "#avg_fanout"] = "1000000000"
                row[analysis_name + "#min_fanout"] = "1000000000"
            else:
                row[analysis_name + "#fanout"] = "1000000000"

        #The remaining attributes are at the graph level,
        # and don't need the edge to be present
        if KEEP_PROGRAM_LEVEL_COLS:
            #row[analysis_name + "#graph_rel_node_count"] = str(
            #    float(graph.relative_node_count))
            #row[analysis_name + "#graph_rel_edge_count"] = str(
            #    float(graph.relative_edge_count))
            row[analysis_name + "#graph_node_count"] = str(
                float(graph.node_count))
            row[analysis_name + "#graph_edge_count"] = str(
                float(graph.edge_count))
            row[analysis_name + "#graph_avg_deg"] = str(
                float(graph.avg_deg))
            row[analysis_name + "#graph_avg_edge_fanout"] = str(
                float(graph.avg_edge_fanout))
            #row[analysis_name + "#graph_num_orphan_nodes"] = str(
            #    float(graph.num_orphan_nodes))
      

def compute_output_imp(row, union_edge, graph: Graph, analysis_name):
    '''Write out the new computed information on edge depths, etc.
    '''
    edge_in_graph = None
    if union_edge.src in graph.nodes: 
        for edge2 in graph.nodes[union_edge.src].edges:
            if (edge2.dest==union_edge.dest
                    and edge2.bytecode_offset==union_edge.bytecode_offset):
                edge_in_graph = edge2
                break
    
    #If the union_edge exists, write attribute values calculated
    if edge_in_graph is not None:
        if edge_in_graph.depth_from_main == UNREACHABLE:
            row[analysis_name + "#depth_from_main"] = "1000000000"
        else:
            row[analysis_name + "#depth_from_main"] = edge_in_graph.depth_from_main
    
        row[analysis_name + "#src_node_in_deg"] = edge_in_graph.src_node_in_deg
        row[analysis_name + "#dest_node_out_deg"] = len(graph.nodes[union_edge.dest].edges)
        row[analysis_name + "#dest_node_in_deg"] = edge_in_graph.dest_node_in_deg
        row[analysis_name + "#src_node_out_deg"] = len(graph.nodes[union_edge.src].edges)
        row[analysis_name + "#repeated_edges"] = edge_in_graph.repeated_edges
        row[analysis_name + "#node_disjoint_paths_from_main"] = edge_in_graph.node_disjoint_paths_from_main
        row[analysis_name + "#edge_disjoint_paths_from_main"] = edge_in_graph.edge_disjoint_paths_from_main
       
        if REMOVE_OFFSETS:
            row[analysis_name + "#avg_fanout"] = edge_in_graph.avg_fanout
            row[analysis_name + "#min_fanout"] = edge_in_graph.min_fanout
        else:
            row[analysis_name + "#fanout"] = edge_in_graph.fanout

    #Else write out the default value - 
    #(because the final table cannot have empty cells)
    else: 
        row[analysis_name + "#depth_from_main"] = "1000000000"
        #row[analysis_name + "#depth_from_orphans"] = "1000000000"
        row[analysis_name + "#src_node_in_deg"] = "-1"
        row[analysis_name + "#dest_node_out_deg"] = "-1"
        row[analysis_name + "#dest_node_in_deg"] = "-1"
        row[analysis_name + "#src_node_out_deg"] = "-1"
        row[analysis_name + "#repeated_edges"] = "0"
        row[analysis_name + "#node_disjoint_paths_from_main"] = "-1"
        row[analysis_name + "#edge_disjoint_paths_from_main"] = "-1"
      
        if REMOVE_OFFSETS:
            row[analysis_name + "#avg_fanout"] = "1000000000"
            row[analysis_name + "#min_fanout"] = "1000000000"
        else:
            row[analysis_name + "#fanout"] = "1000000000"

    #The remaining attributes are at the graph level,
    # and don't need the edge to be present
    if KEEP_PROGRAM_LEVEL_COLS:
        row[analysis_name + "#graph_node_count"] = str(
            float(graph.node_count))
        row[analysis_name + "#graph_edge_count"] = str(
            float(graph.edge_count))
        row[analysis_name + "#graph_avg_deg"] = str(
            float(graph.avg_deg))
        row[analysis_name + "#graph_avg_edge_fanout"] = str(
            float(graph.avg_edge_fanout))

'''  -> Not maintained            
def compute_output_averaged(row,union_edge,callgraphs):
    Write out the new computed information on edge depths, etc, but
    most features will be averaged over all the analyses. This is 
    because several analyses may not have an edge. Hence an 
    individual column for every analysis may not be useful.
    
    edge_depths = []
    src_node_in_deg = []
    dest_node_out_deg = []
    fanouts = []
    source_node_edge_counts = []
    reachable_from_main = 0 #Initial vaue. Represents unreachable. 
    num_paths_to_this = []

    for analysis_name,graph in callgraphs.items():
        #Don't compute anything for the dynamic analysis
        if (DYNAMIC_ANALYSIS_NAME==analysis_name):
            continue

        #Find if the union_edge is in the graph
        edge_in_graph = None
        if union_edge.src in graph.nodes: 
            for edge2 in graph.nodes[union_edge.src].edges:
                if (edge2.dest==union_edge.dest
                        and edge2.bytecode_offset==union_edge.bytecode_offset):
                    edge_in_graph = edge2
                    break

        if edge_in_graph is not None:
            edge_depths.append(edge_in_graph.depth)
            src_node_in_deg.append(edge_in_graph.src_node_in_deg)
            dest_node_out_deg.append(edge_in_graph.dest_node_out_deg)
            fanouts.append(edge_in_graph.fanout)
            source_node_edge_counts.append(
                len(graph.nodes[union_edge.src].edges))
            num_paths_to_this.append(edge_in_graph.num_paths_to_this)
            if edge_in_graph.reachable_from_main:
                reachable_from_main = 1     #represents reachable

        #The remaining attributes are at the graph level,
        # and don't need the edge to be present. Hence won't be averaged
        row[analysis_name + "#graph_rel_node_count"] = str(
            graph.relative_node_count)
        row[analysis_name + "#graph_rel_edge_count"] = str(
            graph.relative_edge_count)

    #Compute, the average values and add it to the row
    if len(edge_depths)>0:
        row["edge_depth"] = round(statistics.mean(edge_depths), 2)
        row["src_node_in_deg"] = round(statistics.mean(src_node_in_deg), 2)
        row["dest_node_out_deg"] = round(statistics.mean(dest_node_out_deg), 2)
        row["fanout"] = round(statistics.mean(fanouts), 2)
        row["src_node_out_deg"] = (
            round(statistics.mean(source_node_edge_counts), 2))
        row["reachable_from_main"] = reachable_from_main
        row["num_paths_to_this"] = round(statistics.mean(num_paths_to_this), 2)

    else: #edge was only present in dynamic analysis. Add default values.
        row["edge_depth"] = "1000000000"
        row["src_node_in_deg"] = "-1"
        row["dest_node_out_deg"] = "-1"
        row["fanout"] = "1000000000"
        row["src_node_out_deg"] = "SOURCE_NODE_DEG_DEFAULT"
        row["reachable_from_main"] = "REACHABLE_FROM_MAIN_DEFAULT"
        row["num_paths_to_this"] = "NUM_PATHS_TO_THIS_DEFAULT"
'''

def get_orphan_nodes(graph):
    '''Returns the set of orphan nodes (those with no incoming edge) 
    in the graph (includes main)
    '''
    orphan_nodes = set(graph.nodes.keys()) #Initialize everyone as orphan
    for node_name,node_object in graph.nodes.items():
        for edge in node_object.edges:
            if edge.dest in orphan_nodes:
                #If a node appears at the end of an edge, it is not an orphan
                orphan_nodes.remove(edge.dest)
    return orphan_nodes

def compute_edge_depths(graph,main_method,orphan_nodes):
    '''Computes the depth of an edge 
    (defined as the depth of the source node) - using BFS
    '''
    node_depths_main = compute_bfs_node_depths(graph,[main_method])
    '''orhpan-depth hard to justify'''
    #if orphan_nodes:
    #    node_depths_orphans = compute_bfs_node_depths(graph,list(orphan_nodes))

    #Now record the edge depths and the depth of the source node of that edge
    for node_name,node_object in graph.nodes.items():
        for edge in node_object.edges:
            edge.depth_from_main = node_depths_main[node_object]
            #if orphan_nodes:
            #    edge.depth_from_orphans = node_depths_orphans[node_object]


def compute_bfs_node_depths(graph: Graph,zero_depth_nodes):
    '''Just a helper function for compute_edge_depths(). The
    only reason for factoring out into a separate function is to
    avoid duplicating code
    '''
    node_depths = {}
    nodes_to_visit = queue.Queue()
    explored_set = set()
    #Initialize every zero_depth_node(root nodes) to depth 0
    #Initialize every other_node to depth -1 (unreachable)
    for node in graph.nodes:
        if node in zero_depth_nodes:
            node_depths[graph.nodes[node]] = 0
            nodes_to_visit.put(node)
            explored_set.add(node)
        else:
            node_depths[graph.nodes[node]] = UNREACHABLE

    #First compute the node depths using BFS
    while (not nodes_to_visit.empty()):
        current_node = nodes_to_visit.get()
        for edge in graph.nodes[current_node].edges:
            if edge.dest not in explored_set:
                explored_set.add(edge.dest)
                nodes_to_visit.put(edge.dest)
                node_depths[graph.nodes[edge.dest]]= (
                    node_depths[graph.nodes[current_node]] + 1)
    return node_depths


def compute_edge_reachability(graph):
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            if edge.depth_from_main!=-1:
                edge.reachable_from_main = True
            else:
                edge.reachable_from_main = False


def compute_src_node_in_deg(graph):
    '''
    For every edge 'e', just increment the 'src_node_in_deg' variable 
    of outgoing edge from the 'e.dest'
    '''
    for node in graph.nodes:
        for incomingedge in graph.nodes[node].edges:
            srcNode = incomingedge.dest
            for edge in graph.nodes[srcNode].edges:
                edge.src_node_in_deg += 1


def compute_dest_node_in_deg(graph):
    '''
    For every node, first compute the in-degree.
    Then every edge which has this as the destination node
    can be updated with it's value
    '''
    #Compute in-degree for each node
    in_degs = {}
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            if edge.dest not in in_degs:
                in_degs[edge.dest] = 0
            in_degs[edge.dest] += 1

    #Update dest_node_in_deg for each edge
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            edge.dest_node_in_deg = in_degs[edge.dest]

def compute_edge_fanouts(graph):
    '''For every edge, compute the number of edges from the same node, 
    with the same bytecode-offset
    '''
    for node in graph.nodes:
        #For 'node', first compute the number of edges at each unique 
        #bytecode-offset. This is accomplished with a Hashtable with 
        #(key=bytecode), and (value = no. of edges with same bytecode offset)
        fanout_hashtable = {} 
        for edge in graph.nodes[node].edges:
            if (edge.bytecode_offset not in fanout_hashtable):
                fanout_hashtable[edge.bytecode_offset] = 0
            fanout_hashtable[edge.bytecode_offset] += 1

        #Now update each edge with the number of edges 
        #at the same bytecode-offset as it
        for edge in graph.nodes[node].edges:
            edge.fanout = fanout_hashtable[edge.bytecode_offset]

        if REMOVE_OFFSETS:
            aggregate_fanouts_hashtable = {}
            for edge in graph.nodes[node].edges:
                if (edge.dest not in aggregate_fanouts_hashtable):
                    aggregate_fanouts_hashtable[edge.dest] = []
                aggregate_fanouts_hashtable[edge.dest].append(edge.fanout)

            for edge in graph.nodes[node].edges:
                edge.min_fanout = min(aggregate_fanouts_hashtable[edge.dest])
                edge.avg_fanout = statistics.mean(
                    aggregate_fanouts_hashtable[edge.dest])



def compute_repeated_edges(graph):
    '''For every edge, compute the number of edges from the same node,
    with the same destination node
    '''
    for node in graph.nodes:
        #For 'node', first compute the number of edges for each unique 
        #dest. This is accomplished with a Hashtable with 
        #(key=dest), and (value = no. of edges with same dest)
        dest_hashtable = {}
        for edge in graph.nodes[node].edges:
            if (edge.dest not in dest_hashtable):
                dest_hashtable[edge.dest] = 0
            dest_hashtable[edge.dest] += 1

        #Now update each edge with the number of edges 
        #with the same dest as it
        for edge in graph.nodes[node].edges:
            edge.repeated_edges = dest_hashtable[edge.dest]

'''The number of outgoing edges of an edge is 
    the number of outgoing edges from it's destination node.
def compute_dest_node_out_deg(graph):
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            edge.dest_node_out_deg = len(graph.nodes[edge.dest].edges)
'''


def compute_node_and_edge_counts(graph):
    '''Simple node and edge counts at the graph level'''
    graph.node_count = len(graph.nodes)
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            graph.edge_count += 1


def compute_relative_node_and_edge_counts(graph,ref_graph):
    '''Computes the node and edge counts in the graph,
    relative to the reference graph'''
    graph.relative_node_count = (
        graph.node_count / ref_graph.node_count)
    graph.relative_edge_count = (
        graph.edge_count / ref_graph.node_count)

def compute_number_of_paths(graph,main_method,orphans):
    '''Computes the number of paths from main and orphan nodes. 
    Uses a Monte-Carlo simulation technique to do this
    estimation. The technique is described in this paper:
    Algorithm 1 of "Estimating the Number of s-t Paths in a Graph"
    by Ben Roberts and Dirk P. Kroese.
    Algorithm 1 is chosen instead of Algorithm 2 since it is 
    biased towards shorter paths, which is what we want.
    (Algorithm Assumption - main must be an orphan node.)

    Note: The original algorithm is for the number of s-t path
    If we enumerate over all nodes 't' and compute the number
    of paths from main, the complexity for each graph is
    = #nodes * lengthOfRandomWalk * #simulationIterations
    = O(n) * O(n) * 10k
    = 10^12 (if n=10^4)
    (* #programs * #analyses for a total count)

    Hence we implement an approximation of this algorithm
    wherein we calculate liklihoods of all nodes along the
    random walk instead of just 1 source node.
    1. Do a random walk without repeating nodes
    2. At each step likelihood = likelihood/node_degree
    3. For each node add 1/likelihood to its total
    '''
    NUMBER_OF_ITERATIONS = 20000 #no of simulation iterations

    #Run the simulation for main, and then orphans to get the 
    #no_of_path_scores for each node.
    no_of_paths_scores_main = num_paths_simulation(
        graph,[main_method],NUMBER_OF_ITERATIONS)
    ''' --- removing the part for orphans because it is 
    not very useful, and it takes too long'''
    #if orphans:
    #    no_of_paths_scores_orphans = num_paths_simulation(
    #        graph,list(orphans),NUMBER_OF_ITERATIONS)
    

    #Now write the computed score for each outgoing edge of a node
    for node_name,node_object in graph.nodes.items():
        for edge in node_object.edges:
            edge.num_paths_to_this_from_main = (
                no_of_paths_scores_main[node_object] / NUMBER_OF_ITERATIONS)
            #if orphans:
            #    edge.num_paths_to_this_from_orphans = (
            #        no_of_paths_scores_orphans[node_object] / NUMBER_OF_ITERATIONS)

 
def num_paths_simulation(graph,starting_nodes,num_iteratations):
    '''Just a helper function for compute_number_of_paths. The
    only reason for factoring out into a separate function is to
    avoid duplicating code
    '''
    no_of_paths_scores = {}
    #Initialize no_of_paths_score as 0 for every node
    for node_name,node_object in graph.nodes.items():
        no_of_paths_scores[node_object] = 0

    #Main simulation loop
    for iter in range(num_iteratations):
        #Set the visited attribute as false for every node
        for node_name,node_object in graph.nodes.items():
            node_object.visited = False
        
        current_node = random.choice(starting_nodes)    
        likelihood = 1.0
        
        #Loop for random walk through graph.
        while(True):
            #Set the current node as visited
            graph.nodes[current_node].visited = True
            no_of_paths_scores[graph.nodes[current_node]] += 1/likelihood

            #Pick the possible next nodes in the random walk
            possible_next_nodes = []
            for edge in graph.nodes[current_node].edges:
                if not graph.nodes[edge.dest].visited:
                    possible_next_nodes.append(edge.dest)

            #If there is no next node, end the random walk ends
            if not possible_next_nodes:
                break

            #Compute the new likelihood, and the next node in the walk.
            likelihood = likelihood / len(possible_next_nodes)
            current_node = random.choice(possible_next_nodes)

    return no_of_paths_scores


def remove_repeated_edges_from_union(union_edge_set):
    '''Removes all repeated edges in the graph.
    if e1 and e2 are edges with same src, dest and different bytecode
    offset, they are repeated edges. 1 of them will be removed.
    '''
    # unique_src_dest_pairs = set()
    # edges_to_remove = []
    # for edge in union_edge_set:
    #     if (edge.src,edge.dest) in unique_src_dest_pairs: #repeated edge
    #         edges_to_remove.append(edge)
    #     else:
    #         unique_src_dest_pairs.add((edge.src,edge.dest))

    # for edge in edges_to_remove:
    #     union_edge_set.remove(edge)

    unique_src_dest_pairs = set()
    new_union_edge_set = set()
    for edge in union_edge_set:
        if (edge.src,edge.dest) not in unique_src_dest_pairs: 
            unique_src_dest_pairs.add((edge.src,edge.dest))
            new_union_edge_set.add(edge)
            
    union_edge_set = list(new_union_edge_set)

def compute_node_disjoint_paths(graph: Graph, main_method: str, orphan_nodes):
    '''Computes the number of maximal (not the same as maximum) 
    node-disjoint paths (this is an estimate of the
    maximum node disjoint paths) to each edge,
    starting at main, and starting at an orphan node
    '''
    graph_edges = [(s, e.dest) for s, t in graph.nodes.items() for e in t.edges]
    graph_nodes = set([e for t in graph_edges for e in t])
    if '<boot>' not in graph_nodes:
        graph_nodes.add("<boot>")
    label_to_index = {label: index for index, label in enumerate(graph_nodes)}
    G = nk.Graph(len(graph_nodes), directed=False)
    for s, t in graph_edges:
        G.addEdge(label_to_index[s], label_to_index[t])

    for node in graph.nodes:
        if node==main_method or (node in orphan_nodes):
            #print(node)
            continue #skip main method and orphan nodes
        node_disjoint_paths_from_main = 0
        #nodes_used_up = set()
        #Each loop iteration looks for 1 path, 
        #and then removes the nodes on that path
        # while True:
        #     path = find_node_disjoint_path(graph,main_method,node,nodes_used_up)
        #     if path==None: #No more paths exist
        #         break
        #     for n in path:
        #         nodes_used_up.add(n)
        #     node_disjoint_paths_from_main += 1
        
        bfs = nk.distance.BFS(G, source=label_to_index[main_method], target=label_to_index[node]).run()
        node_disjoint_paths_from_main = bfs.numberOfPaths(label_to_index[node])
        #Add the calculated value to every outgoing edge from this node.
        for edge in graph.nodes[node].edges:
            edge.node_disjoint_paths_from_main = node_disjoint_paths_from_main

    #For edges out of main and orphans set the number of node disjoint paths as 1
    for edge in graph.nodes[main_method].edges:
        edge.node_disjoint_paths_from_main = 1
    
    #for orphan in orphan_nodes:
    #    for edge in graph.nodes[orphan].edges:
    #        edge.node_disjoint_paths_from_orphans = 1

def find_node_disjoint_path(graph,start_node,dest_node,nodes_used_up):
    '''This function finds a path from the current node to 
    the destination node using BFS
    '''
    #Initialization for BFS
    nodes_to_visit = deque()
    nodes_to_visit.append(start_node)
    explored_set = {start_node}
    parent_node = {}
    for node in graph.nodes:
        parent_node[node] = None

    #Main BFS-loop
    while nodes_to_visit:  # empty deques are "falsy"
        current_node = nodes_to_visit.popleft()
        for edge in graph.nodes[current_node].edges:
            #If dest_node was found
            if current_node==dest_node:
                #Return the path by retracing the parent pointers
                path_to_dest = []
                while current_node is not None:
                    path_to_dest.append(current_node)
                    current_node = parent_node[current_node]
                return path_to_dest
            #Add the edge to the set of nodes to visit,if it has not 
            #already been explored, and is not in 'nodes_used_up'
            if (edge.dest not in explored_set 
                    and edge.dest not in nodes_used_up):
                explored_set.add(edge.dest)
                parent_node[edge.dest] = current_node
                nodes_to_visit.append(edge.dest)
    return None #No path found


# NOTE: Super slow for large graphs
# NOTE: Gives SystemERROR when using queue.Queue()
# def find_node_disjoint_path(graph,start_node,dest_node,nodes_used_up):
#     '''This function finds a path from the current node to 
#     the destination node using BFS
#     '''
#     #Initialization for BFS
#     nodes_to_visit = queue.Queue()
#     nodes_to_visit.put(start_node)
#     explored_set = {start_node}
#     parent_node = {}
#     for node in graph.nodes:
#         parent_node[node] = None

#     #Main BFS-loop
#     while (not nodes_to_visit.empty()):
#         current_node = nodes_to_visit.get()
#         for edge in graph.nodes[current_node].edges:
#             #If dest_node was found
#             if current_node==dest_node:
#                 #Return the path by retracing the parent pointers
#                 path_to_dest = []
#                 while current_node!=None:
#                     path_to_dest.append(current_node)
#                     current_node = parent_node[current_node]
#                 return path_to_dest
#             #Add the edge to the set of nodes to visit,if it has not 
#             #already been explored, and is not in 'nodes_used_up'
#             if (edge.dest not in explored_set 
#                     and edge.dest not in nodes_used_up):
#                 explored_set.add(edge.dest)
#                 parent_node[edge.dest] = current_node
#                 nodes_to_visit.put(edge.dest)
#     return None #No path found


def compute_edge_disjoint_paths(graph,main_method,orphan_nodes):
    '''Computes the number of maximal (not the same as maximum) 
    edge-disjoint paths (this is an estimate of the
    maximum node disjoint paths) to each edge,
    starting at main, and starting at an orphan node
    '''

    #For large graphs, skip this function because it takes too long
    edges_in_graph = 0
    for node in graph.nodes:
        edges_in_graph += len(graph.nodes[node].edges)
    
    if (edges_in_graph>LARGE_GRAPH_CUTOFF):
        for node in graph.nodes:
            for edge in graph.nodes[node].edges:
                edge.edge_disjoint_paths_from_main = -1
    else:
        for node in graph.nodes:
            if node==main_method or (node in orphan_nodes):
                continue #skip main method and orphan nodes
            edge_disjoint_paths_from_main = 0
            #Set of remaining edges. Every time a path is found,
            #the edges on the path are deleted from here
            edges_left = {} 
            for n in graph.nodes:
                edges_left[n] = set(graph.nodes[n].edges)
            #Each loop iteration looks for 1 path, 
            #and then removes the edges on that path
            while True:
                path = find_edge_disjoint_path(graph,main_method,node,edges_left)
                if path==None: #No more paths exist
                    break
                for (n,e) in path:
                    edges_left[n].remove(e)
                edge_disjoint_paths_from_main += 1
            #Add the calculated value to every outgoing edge from this node.
            for edge in graph.nodes[node].edges:
                edge.edge_disjoint_paths_from_main = edge_disjoint_paths_from_main

        #For edges out of main set the number of node disjoint paths as 1
        for edge in graph.nodes[main_method].edges:
            edge.edge_disjoint_paths_from_main = 1
        
        '''
        -> this part takes way too long
        #Now repeat a similar (but not the same) procedure as above
        #for the orphan nodes
        edge_disjoint_paths_from_orphans = 0
        edges_left = {} 
        for n in graph.nodes:
            edges_left[n] = set(graph.nodes[n].edges)
        #Assuming from each orphan we can find at most 1 edge_disjoint_path.
        #Also the 'edges_left' is common among all the iterations.
        for orphan in orphan_nodes:
            path = find_edge_disjoint_path(graph,orphan,node,edges_left)
            if path!=None:
                for (n,e) in path:
                    edges_left[n].remove(e)
                edge_disjoint_paths_from_orphans += 1
        #Add the calculated value to every outgoing edge from this node.
        for edge in graph.nodes[node].edges:
            edge.edge_disjoint_paths_from_orphans = edge_disjoint_paths_from_orphans
        '''

# def process_node(node, graph, main_method, orphan_nodes):
#     if node == main_method or (node in orphan_nodes):
#         return None # skip main method and orphan nodes
#     edge_disjoint_paths_from_main = 0
#     # Set of remaining edges. Every time a path is found,
#     # the edges on the path are deleted from here
#     edges_left = {} 
#     for n in graph.nodes:
#         edges_left[n] = set(graph.nodes[n].edges)
#     # Each loop iteration looks for 1 path, 
#     # and then removes the edges on that path
#     while True:
#         path = find_edge_disjoint_path(graph, main_method, node, edges_left)
#         if path is None: # No more paths exist
#             break
#         for (n, e) in path:
#             edges_left[n].remove(e)
#         edge_disjoint_paths_from_main += 1
#     # Add the calculated value to every outgoing edge from this node.
#     for edge in graph.nodes[node].edges:
#         edge.edge_disjoint_paths_from_main = edge_disjoint_paths_from_main
#     return node, graph.nodes[node].edges

def find_edge_disjoint_paths(parent_node_and_edge, main_method, node: Node, edges_left):
    edge_disjoint_paths_from_main = 0
    while True:
        path = find_edge_disjoint_path_opt(parent_node_and_edge, main_method, node, edges_left)
        if path is None: # No more paths exist
            break
        for (n, e) in path:
            edges_left[n].remove(e)
        edge_disjoint_paths_from_main += 1
    return node, edge_disjoint_paths_from_main

def batch_edge_disjoint_paths(nodes_batch, graph, main_method, orphan_nodes):
    results = []
    for node in nodes_batch:
        if node==main_method or (node in orphan_nodes):
            continue #skip main method and orphan nodes
        #Set of remaining edges. Every time a path is found,
        #the edges on the path are deleted from here
        edges_left = {} 
        for n in graph.nodes:
            edges_left[n] = set(graph.nodes[n].edges)

        parent_node_and_edge = {} #need to record parent edge as well as node
        for n in graph.nodes:
            parent_node_and_edge[n] = (None,None)

        result = find_edge_disjoint_paths(parent_node_and_edge, copy.copy(main_method), copy.copy(node), edges_left)
        if result is not None:
            node, edge_disjoint_paths_from_main = result
            results.append((node, edge_disjoint_paths_from_main))
    return results

def compute_edge_disjoint_paths_parallel(graph, main_method: str, orphan_nodes):
    edges_in_graph = 0
    for node in graph.nodes:
        edges_in_graph += len(graph.nodes[node].edges)

    if (edges_in_graph > LARGE_GRAPH_CUTOFF):
        for node in graph.nodes:
            for edge in graph.nodes[node].edges:
                edge.edge_disjoint_paths_from_main = -1
    else:
        # with ProcessPoolExecutor(max_workers=24) as executor:
        #     futures = []
        #     print(f"No. of nodes {len(graph.nodes)}")
        #     for node in graph.nodes:
        #         if node==main_method or (node in orphan_nodes):
        #             continue #skip main method and orphan nodes
        #         #Set of remaining edges. Every time a path is found,
        #         #the edges on the path are deleted from here
        #         edges_left = {} 
        #         for n in graph.nodes:
        #             edges_left[n] = set(graph.nodes[n].edges)

        #         parent_node_and_edge = {} #need to record parent edge as well as node
        #         for n in graph.nodes:
        #             parent_node_and_edge[n] = (None,None)

        #         futures.append(executor.submit(find_edge_disjoint_paths, parent_node_and_edge, copy.copy(main_method), copy.copy(node), edges_left))
        #         #futures = {executor.submit(process_node, node, graph, main_method, orphan_nodes): node for node in graph.nodes}
        #     for future in futures:
        #         result = future.result()
        #         if result is not None:
        #             node, edge_disjoint_paths_from_main = result
        #             for edge in graph.nodes[node].edges:
        #                 edge.edge_disjoint_paths_from_main = edge_disjoint_paths_from_main
        with ProcessPoolExecutor(max_workers=24) as executor:
            futures = []
            print(f"No. of nodes {len(graph.nodes)}")
            
            batch_size = 500
            nodes_batches = [list(graph.nodes)[i:i+batch_size] for i in range(0, len(graph.nodes), batch_size)]
            
            for nodes_batch in nodes_batches:
                futures.append(executor.submit(batch_edge_disjoint_paths, nodes_batch, graph, main_method, orphan_nodes))
                
            for future in futures:
                results = future.result()
                if results is not None:
                    for node, edge_disjoint_paths_from_main in results:
                        for edge in graph.nodes[node].edges:
                            edge.edge_disjoint_paths_from_main = edge_disjoint_paths_from_main

        # For edges out of main set the number of node disjoint paths as 1
        for edge in graph.nodes[main_method].edges:
            edge.edge_disjoint_paths_from_main = 1


def find_edge_disjoint_path_opt(parent_node_and_edge, start_node, dest_node, edges_left):
    '''This function finds a path from the current node to 
    the destination node using BFS
    '''
    #Initialization for BFS
    nodes_to_visit = deque()
    nodes_to_visit.append(start_node)
    explored_set = {start_node}

    #Main BFS-loop
    while nodes_to_visit:
        current_node = nodes_to_visit.popleft()
        #If dest_node was found
        if current_node == dest_node:
            #Return the path by retracing the parent pointers
            path_to_dest = []
            current_node,current_edge = parent_node_and_edge[dest_node]
            while current_node is not None:
                path_to_dest.append((current_node,current_edge))
                current_node,current_edge = parent_node_and_edge[current_node]
            return path_to_dest
        #Else continue BFS
        for edge in edges_left[current_node]:
            #Add the edge to the set of nodes to visit, if it has not 
            #already been explored.
            if edge.dest not in explored_set:
                explored_set.add(edge.dest)
                parent_node_and_edge[edge.dest] = (current_node,edge)
                nodes_to_visit.append(edge.dest)
    return None #No path found


def find_edge_disjoint_path(graph,start_node,dest_node,edges_left):
    '''This function finds a path from the current node to 
    the destination node using BFS
    '''
    #Initialization for BFS
    nodes_to_visit = queue.Queue()
    nodes_to_visit.put(start_node)
    explored_set = {start_node}
    parent_node_and_edge = {} #need to record parent edge as well as node
    for node in graph.nodes:
        parent_node_and_edge[node] = (None,None)

    #Main BFS-loop
    while (not nodes_to_visit.empty()):
        current_node = nodes_to_visit.get()
        #If dest_node was found
        if current_node==dest_node:
            #Return the path by retracing the parent pointers
            path_to_dest = []
            current_node,current_edge = (
                    parent_node_and_edge[dest_node])
            while current_node!=None:
                path_to_dest.append((current_node,current_edge))
                current_node,current_edge = (
                    parent_node_and_edge[current_node])
            return path_to_dest
        #Else continue BFS
        for edge in edges_left[current_node]:
            #Add the edge to the set of nodes to visit, if it has not 
            #already been explored.
            if edge.dest not in explored_set:
                explored_set.add(edge.dest)
                parent_node_and_edge[edge.dest] = (current_node,edge)
                nodes_to_visit.put(edge.dest)
    return None #No path found

def compute_graph_level_info(graph,orphan_nodes):
    '''Compute some graph level information.
    Will be common to all edges.
    '''
    #Compute the number of orphan nodes
    graph.num_orphan_nodes = len(orphan_nodes)    

    #Compute the average degree of the nodes
    total_deg = 0
    total_nodes = 0
    for node_name,node_object in graph.nodes.items():
        total_nodes += 1.0
        total_deg += len(node_object.edges)
    graph.avg_deg = total_deg/total_nodes

    #Compute average edge fanout
    total_fanout = 0
    total_edges = 0
    for node_name,node_object in graph.nodes.items():
        for edge in node_object.edges:
            total_edges += 1.0
            total_fanout += edge.fanout
    graph.avg_edge_fanout = total_fanout/total_edges

def main():
    #Loop through all the file names
    for testcase in pathlib.Path(BENCHMARKS_FOLDER).iterdir():
        if not testcase.is_dir(): #skip non-directories
            continue
        if (testcase / OUTPUT_DATASET_FILE).is_file():
            print("Testcase: " + testcase.name + " - output already exists")
            continue #skip if output already exists
        #Progress information
        print("Testcase: " + testcase.name)
        #Read the combination file
        with open(testcase / DATASET_FILE, "r") as readfp:  
            #Some initialization
            callgraphs = {} #The dictionary of graphs for each of the analyses
            union_edge_set = []

            #Get the names of the analyses
            csv_reader = csv.DictReader(readfp)
            analysis_names = csv_reader.fieldnames[3:]

            #Create a graph for each analysis
            for analysis in analysis_names:
                callgraphs[analysis] = Graph()

            #Read rest of file
            for row in csv_reader:
                #Add the edge to the union call graph
                union_edge_set.append(
                    UnionEdge(row['method'],row['offset'],row['target']))

                #Loop through the 0-1 bits for each analysis
                for analysis in analysis_names:
                    #if true, then add the edge to the respective graph. 
                    #Else do nothing.
                    if row[analysis]=='1': 
                        #Create new node if it doesn't exist. Then add edge
                        if (row['method'] not in callgraphs[analysis].nodes):
                            callgraphs[analysis].nodes[row['method']] = Node()
                        if (row['target'] not in callgraphs[analysis].nodes):
                            callgraphs[analysis].nodes[row['target']] = Node()
                        callgraphs[analysis].nodes[row['method']].edges.add(
                            Edge(row['offset'],row['target']))

            '''
            #Read the main classname, and compute the main method name
            with open(testcase / BENCHMARK_INFO_FILE) as filep:
                mainclass = json.load(filep)["mainclass"]
                main_method = (mainclass.replace(".", "/") 
                    + ".main:([Ljava/lang/String;)V")
            '''
            main_method = "<boot>"

            #Get the node and edge counts for each graph
            for analysis_name,graph in callgraphs.items():
                compute_node_and_edge_counts(graph)

            #For each analysis, 
            for analysis_name,graph in callgraphs.items():
                #compute the set of orphan nodes
                orphan_nodes = get_orphan_nodes(graph)
                if main_method in orphan_nodes: orphan_nodes.remove(main_method)
                #compute the depth of each edge
                compute_edge_depths(graph,main_method,orphan_nodes)
                #compute the edge reachability information
                compute_edge_reachability(graph)
                #compute the no. of incoming edges for each edge
                compute_src_node_in_deg(graph)
                #compute the no. of outgoing edges for each edge
                #compute_dest_node_out_deg(graph) =len(dest_node.edges) - hence skipped
                #compute the no. of incoming edges of destination node
                compute_dest_node_in_deg(graph)
                #compute the fanout for each edge
                compute_edge_fanouts(graph)
                #compute the node and edge counts, 
                #as ratios to the size of the 'REFERENCE_ANALYSIS' 
                # -> can't be explained compute_relative_node_and_edge_counts(
                #    graph,callgraphs[REFERENCE_ANALYSIS])
                #compute the number of paths from main 
                #and from orphan nodes
                #not having any feature imp 
                #-> compute_number_of_paths(graph,main_method,orphan_nodes)
                #compute the following for each edge 'e'
                # the number of edges with same source node and dest as 'e'
                compute_repeated_edges(graph)
                #compute the number of node-disjoint paths from main, and from
                #the set of orphan nodes
                compute_node_disjoint_paths(graph,main_method,orphan_nodes)
                #compute the number of edge-disjoint paths from main, and from
                #the set of orphan nodes
                compute_edge_disjoint_paths(graph,main_method,orphan_nodes)
                #Record number of nodes, edges, orphans
                compute_graph_level_info(graph,orphan_nodes)

            #For the union edge set, remove repeated edges (edges with
            #same src, dest but different offset)
            if REMOVE_OFFSETS:
                remove_repeated_edges_from_union(union_edge_set)

            #Write output
            with open(testcase / OUTPUT_DATASET_FILE, "w") as fp:
                write_output(fp,csv_reader,union_edge_set,callgraphs)


def process_program(cg_df: pd.DataFrame):
    proj = cg_df['program_name'].iloc[0]
    main_method = "<boot>"
    #cg_df = static_dyn_common[proj]
    #cg_df = xcorpus_df[xcorpus_df['program_name'] == proj]

    union_edge_set = []
    call_graphs_w_feat = Graph()

    for i, r in cg_df.iterrows():
        union_edge_set.append(UnionEdge(r['method'], r['offset'], r['target'], r['wiretap'],
                                        r['wala-cge-0cfa-noreflect-intf-trans']))
        if r['method'] not in call_graphs_w_feat.nodes:
            call_graphs_w_feat.nodes[r['method']] = Node()
        if r['target'] not in call_graphs_w_feat.nodes:
            call_graphs_w_feat.nodes[r['target']] = Node()
        call_graphs_w_feat.nodes[r['method']].edges.add(Edge(r['offset'], r['target']))

    compute_node_and_edge_counts(call_graphs_w_feat)
    orphan_nodes = get_orphan_nodes(call_graphs_w_feat)

    if main_method in orphan_nodes:
        orphan_nodes.remove(main_method)

    compute_edge_depths(call_graphs_w_feat, main_method, orphan_nodes)
    #print(f"Computed edge depths for project: {proj}")
    compute_edge_reachability(call_graphs_w_feat)
    #print(f"Computed edge reachability for project: {proj}")
    compute_src_node_in_deg(call_graphs_w_feat)
    #print(f"Computed source node in-degrees for project: {proj}")
    compute_dest_node_in_deg(call_graphs_w_feat)
    #print(f"Computed destination node in-degrees for project: {proj}")
    compute_edge_fanouts(call_graphs_w_feat)
    #print(f"Computed edge fanouts for project: {proj}")
    compute_repeated_edges(call_graphs_w_feat)
    #print(f"Computed repeated edges for project: {proj}")
    compute_node_disjoint_paths(call_graphs_w_feat, main_method, orphan_nodes)
    #print(f"Computed node disjoint paths for project: {proj}")
    compute_edge_disjoint_paths(call_graphs_w_feat, main_method, orphan_nodes)
    #print(f"Computed edge disjoint paths for project: {proj}")
    compute_graph_level_info(call_graphs_w_feat, orphan_nodes)
    #print(f"Computed graph level information for project: {proj}")
    remove_repeated_edges_from_union(union_edge_set)
    #print(f"Removed repeated edges for project: {proj}")

    edge_samples = []
    for union_edge in union_edge_set:
        row_sample = {}
        add_old_entries_to_row_imp(row_sample, union_edge)
        compute_output_imp(row_sample, union_edge, call_graphs_w_feat,
                                          'wala-cge-0cfa-noreflect-intf-trans')
        edge_samples.append(row_sample)
    edge_samples_df = pd.DataFrame.from_dict(edge_samples)
    return edge_samples_df

def add_struct_feat(prog_df: pd.DataFrame, cg_df: pd.DataFrame):
    for i, r in tqdm(prog_df.iterrows(), total=len(prog_df), disable=True):
        # xcorpus_df = xcorpus_programs_feat[r['program_name']]
        # match_r = xcorpus_programs_df_idx[r['program_name']][r['method']+"|"+r['target']]
        for f in struct_feat_names:
            prog_df.at[i, f] = cg_df.iloc[i][f]
    return prog_df

def gen_struct_feat(prog_df: pd.DataFrame) -> pd.DataFrame:
    prog_df_struct_w_feat = process_program(prog_df)
    prog_df = add_struct_feat(prog_df, prog_df_struct_w_feat)
    scaler = MinMaxScaler()
    for f in struct_feat_names:
       prog_df[f] = scaler.fit_transform(prog_df[f].to_numpy().reshape(-1, 1))
    return prog_df

if __name__ == '__main__':
    main()
