"""Methods for computing ngram counts"""
import copy
import itertools
import typing
from collections import Counter, defaultdict, deque

import networkx as nx

from .jsgf_graph import get_start_end_nodes


def get_intent_ngram_counts(
    graph: nx.DiGraph, pad_start="<s>", pad_end="</s>", order=3
) -> typing.Dict[str, Counter]:
    intent_counts = defaultdict(Counter)
    start_node, end_node = get_start_end_nodes(graph)
    word_graph = to_word_graph(
        graph, start_node, end_node, pad_start=pad_start, pad_end=pad_end
    )

    intent_nodes = set(graph.successors(start_node))
    for intent_node in intent_nodes:
        # __label__INTENT
        olabel = graph.edges[(start_node, intent_node)]["olabel"]
        assert olabel.startswith("__label__"), "Not an intent graph"
        intent_name = olabel[9:]

        # First word(s) of intent
        valid_nodes = set([start_node])
        for word_node in graph.successors(intent_node):
            valid_nodes.add(word_node)
            valid_nodes.update(nx.descendants(word_graph, word_node))

        # Filter out nodes not part of this intent
        def filter_node(n):
            return n in valid_nodes

        # Compute ngram counts using a view of the main word graph restricted to
        # nodes from this intent.
        subgraph = nx.subgraph_view(word_graph, filter_node=filter_node)
        intent_counts[intent_name] = count_ngrams(
            subgraph,
            start_node,
            end_node,
            pad_start=pad_start,
            pad_end=pad_end,
            order=order,
        )

    return intent_counts


# -----------------------------------------------------------------------------


def count_ngrams(
    word_graph: nx.DiGraph,
    start_node: int,
    end_node: int,
    pad_start="<s>",
    pad_end="</s>",
    label="word",
    order=3,
) -> Counter:
    """Compute n-gram counts in a word graph."""
    assert order > 0, "Order must be greater than zero"
    n_data = word_graph.nodes(data=True)

    # Counts from a node to <s>
    up_counts = Counter()

    # Counts from a node to </s>
    down_counts = Counter()

    # Top/bottom = 1
    up_counts[start_node] = 1
    down_counts[end_node] = 1

    # Skip start node
    for n in itertools.islice(nx.topological_sort(word_graph), 1, None):
        for n2 in word_graph.predecessors(n):
            up_counts[n] += up_counts[n2]

    # Down
    reverse_graph = nx.reverse_view(word_graph)

    # Skip end node
    for n in itertools.islice(nx.topological_sort(reverse_graph), 1, None):
        for n2 in reverse_graph.predecessors(n):
            down_counts[n] += down_counts[n2]

    # Compute counts
    ngram_counts = Counter()
    for n in word_graph:
        # Unigram
        word = n_data[n][label]
        ngram = [word]
        ngram_counts[tuple(ngram)] += up_counts[n] * down_counts[n]

        if order == 1:
            continue

        # Higher order
        q = deque([(n, ngram)])
        while q:
            current_node, current_ngram = q.popleft()
            for n2 in word_graph.predecessors(current_node):
                word_n2 = n_data[n2][label]
                ngram_n2 = [word_n2] + current_ngram
                ngram_counts[tuple(ngram_n2)] += up_counts[n2] * down_counts[n]

                if len(ngram_n2) < order:
                    q.append((n2, ngram_n2))

    return ngram_counts


# -----------------------------------------------------------------------------


def to_word_graph(
    graph: nx.DiGraph,
    start_node: int,
    end_node: int,
    pad_start: str = "<s>",
    pad_end: str = "</s>",
    label: str = "word",
) -> nx.DiGraph:
    """Converts a JSGF graph with meta nodes to just words."""

    # Deep copy graph to avoid mutating the original
    graph = copy.deepcopy(graph)
    n_data = graph.nodes(data=True)

    # Add <s> and </s>
    n_data[start_node][label] = pad_start
    n_data[end_node][label] = pad_end

    for node in list(graph):
        word = n_data[node].get(label, "")
        if not word:
            # Clip meta (non-word) node
            for pred_node in graph.predecessors(node):
                for succ_node in graph.successors(node):
                    graph.add_edge(pred_node, succ_node)

            graph.remove_node(node)

    return graph
