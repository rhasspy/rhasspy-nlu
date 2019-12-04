"""Recognition functions for sentences using JSGF graphs."""
import typing

import networkx as nx

# -----------------------------------------------------------------------------


def recognize_strict(
    tokens: typing.List[str], graph: nx.MultiDiGraph
) -> typing.List[str]:
    """Match a single path from the graph exactly if possible."""
    # node -> attrs
    n_data = graph.nodes(data=True)

    # start state
    start_node = [n for n, data in n_data if data.get("start", False)][0]

    # Do breadth-first search
    node_queue = [(start_node, [], tokens)]
    while node_queue:
        current_node, current_path, next_tokens = node_queue.pop()
        is_final = n_data[current_node].get("final", False)
        if is_final:
            # Reached final state
            return current_path

        for next_node, edges in graph[current_node].items():
            for _, edge_data in edges.items():
                next_path = list(current_path)
                ilabel = edge_data.get("ilabel", "")
                olabel = edge_data.get("olabel", "")

                if ilabel and next_tokens:
                    # Failed to match input label
                    if ilabel != next_tokens[0]:
                        continue

                    # Token match
                    next_tokens.pop(0)

                if olabel:
                    # Only add non-empty output label
                    next_path.append(olabel)

                # Continue search
                node_queue.append((next_node, next_path, list(next_tokens)))

    return []


# -----------------------------------------------------------------------------


def recognize_fuzzy(
    tokens: typing.List[str],
    graph: nx.MultiDiGraph,
    stop_words: typing.Optional[typing.Set[str]] = None,
) -> typing.Dict[str, typing.Tuple[typing.List[str], int]]:
    """Do less strict matching using a cost function and optional stop words."""
    stop_words: Set[str] = stop_words or set()

    # node -> attrs
    n_data = graph.nodes(data=True)

    # start state
    start_node: int = [n for n, data in n_data if data.get("start", False)][0]

    # intent -> (symbols, cost)
    intent_symbols_and_costs = {}

    # Lowest cost so far
    best_cost: float = float(len(n_data))

    # (node, in_tokens, out_tokens, cost, intent_name)
    node_queue = [(start_node, tokens, [], 0.0, None)]

    # BFS it up
    while node_queue:
        q_node, q_in_tokens, q_out_tokens, q_cost, q_intent = node_queue.pop(0)
        is_final = n_data[q_node].get("final", False)

        # Update best intent cost on final state.
        # Don't bother reporting intents that failed to consume any tokens.
        if is_final and (q_cost < len(tokens)):
            best_intent_cost = intent_symbols_and_costs.get(q_intent, (None, None))[1]
            final_cost = q_cost + len(q_in_tokens)  # remaning tokens count against

            if (best_intent_cost is None) or (final_cost < best_intent_cost):
                intent_symbols_and_costs[q_intent] = [q_out_tokens, final_cost]

            if final_cost < best_cost:
                best_cost = final_cost

        if q_cost > best_cost:
            continue

        # Process child edges
        for next_node, edges in graph[q_node].items():
            for _, edge_data in edges.items():
                in_label = edge_data.get("ilabel", "")
                out_label = edge_data.get("olabel", "")
                next_in_tokens = list(q_in_tokens)
                next_out_tokens = list(q_out_tokens)
                next_cost = q_cost
                next_intent = q_intent

                if out_label.startswith("__label__"):
                    next_intent = out_label[9:]

                if in_label:
                    if next_in_tokens and (in_label == next_in_tokens[0]):
                        # Consume matching token immediately
                        next_in_tokens.pop(0)

                        if out_label:
                            next_out_tokens.append(out_label)
                    else:
                        # Consume non-matching tokens and increase cost
                        # unless stop word.
                        while next_in_tokens and (in_label != next_in_tokens[0]):
                            bad_token = next_in_tokens.pop(0)
                            if bad_token not in stop_words:
                                next_cost += 1
                            else:
                                # Need a non-zero cost for stop words to
                                # avoid case where two FST paths are
                                # identical, save for stop words.
                                next_cost += 0.1

                        if next_in_tokens:
                            # Consume matching token
                            next_in_tokens.pop(0)

                            if out_label:
                                next_out_tokens.append(out_label)
                        else:
                            # No matching token
                            next_cost += 2
                else:
                    # Consume out label
                    if out_label:
                        next_out_tokens.append(out_label)

                node_queue.append(
                    (next_node, next_in_tokens, next_out_tokens, next_cost, next_intent)
                )

    return intent_symbols_and_costs
