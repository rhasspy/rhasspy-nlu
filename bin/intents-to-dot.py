#!/usr/bin/env python3
import sys

import networkx
import rhasspynlu

print("Reading graph from stdin...", file=sys.stderr)
graph = rhasspynlu.gzip_pickle_to_graph(sys.stdin.buffer)
networkx.drawing.nx_pydot.write_dot(graph, sys.stdout)
