"""Recognition functions for sentences using JSGF graphs."""
from collections import defaultdict
import typing

import attr
import networkx as nx

from .intent import Entity, Intent, Recognition, RecognitionResult, TagInfo

# -----------------------------------------------------------------------------


def recognize(
    tokens: typing.List[str],
    graph: nx.DiGraph,
    fuzzy: bool = True,
    stop_words: typing.Optional[typing.Set[str]] = None,
    **fuzzy_args
) -> typing.Tuple[RecognitionResult, typing.List[Recognition]]:
    if fuzzy:
        # Fuzzy recognition
        best_fuzzy = best_fuzzy_cost(
            paths_fuzzy(tokens, graph, stop_words=stop_words, **fuzzy_args)
        )

        if best_fuzzy:
            recognitions = []

            # Gather all successful fuzzy paths
            for fuzzy_result in best_fuzzy:
                result, recognition = path_to_recognition(
                    fuzzy_result.node_path, graph, cost=fuzzy_result.cost
                )
                if result == RecognitionResult.SUCCESS:
                    recognitions.append(recognition)

            if recognitions:
                return RecognitionResult.SUCCESS, recognitions
    else:
        # Strict recognition
        path = path_strict(tokens, graph)
        if (not path) and stop_words:
            # Try again by excluding stop words
            tokens = [t for t in tokens if t not in stop_words]
            path = path_strict(tokens, graph, exclude_tokens=stop_words)

        result, recognition = path_to_recognition(path, graph)
        if result == RecognitionResult.SUCCESS:
            return result, [recognition]

    # No results
    return RecognitionResult.FAILURE, []


# -----------------------------------------------------------------------------


def path_strict(
    tokens: typing.List[str],
    graph: nx.DiGraph,
    exclude_tokens: typing.Optional[typing.Set[str]] = None,
) -> typing.List[int]:
    """Match a single path from the graph exactly if possible."""
    # node -> attrs
    n_data = graph.nodes(data=True)

    # start state
    start_node: int = [n for n, data in n_data if data.get("start", False)][0]

    # Do breadth-first search
    node_queue = [(start_node, [], tokens)]
    while node_queue:
        current_node, current_path, current_tokens = node_queue.pop(0)
        is_final = n_data[current_node].get("final", False)
        if is_final and (not current_tokens):
            # Reached final state
            return current_path

        for next_node, edge_data in graph[current_node].items():
            next_path = list(current_path)
            next_tokens = list(current_tokens)

            ilabel = edge_data.get("ilabel", "")

            if ilabel:
                if next_tokens:
                    # Failed to match input label
                    if ilabel != next_tokens[0]:
                        if (not exclude_tokens) or (ilabel not in exclude_tokens):
                            # Can't exclude
                            continue
                    else:
                        # Token match
                        next_tokens.pop(0)
                else:
                    # Ran out of tokens
                    continue

            next_path.append(current_node)

            # Continue search
            node_queue.append((next_node, next_path, next_tokens))

    return []


# -----------------------------------------------------------------------------


@attr.s
class FuzzyResult:
    """Single path for fuzzy recognition."""

    intent_name: str = attr.ib()
    node_path: typing.Iterable[int] = attr.ib()
    cost: float = attr.ib()


def paths_fuzzy(
    tokens: typing.List[str],
    graph: nx.DiGraph,
    stop_words: typing.Optional[typing.Set[str]] = None,
    mismatched_word_cost: float = 1,
    stop_word_cost: float = 0.1,
    extra_word_cost: float = 1,
) -> typing.Dict[str, typing.List[FuzzyResult]]:
    """Do less strict matching using a cost function and optional stop words."""
    if not tokens:
        return {}

    stop_words: Set[str] = stop_words or set()

    # node -> attrs
    n_data = graph.nodes(data=True)

    # start state
    start_node: int = [n for n, data in n_data if data.get("start", False)][0]

    # intent -> [(symbols, cost), (symbols, cost)...]
    intent_symbols_and_costs: typing.Dict[str, typing.List[FuzzyResult]] = defaultdict(
        list
    )

    # Lowest cost so far
    best_cost: float = float(len(n_data))

    # (node, in_tokens, out_nodes, out_count, cost, intent_name)
    node_queue = [(start_node, tokens, [], 0.0, 0, None)]

    # BFS it up
    while node_queue:
        q_node, q_in_tokens, q_out_nodes, q_out_count, q_cost, q_intent = node_queue.pop(
            0
        )
        is_final = n_data[q_node].get("final", False)

        # Update best intent cost on final state.
        # Don't bother reporting intents that failed to consume any tokens.
        if is_final and (q_cost < q_out_count):
            best_intent_cost: typing.Optional[float] = None
            best_intent_costs = intent_symbols_and_costs.get(q_intent)
            if best_intent_costs:
                best_intent_cost = best_intent_costs[0].cost

            final_cost = q_cost + len(q_in_tokens)  # remaning tokens count against
            final_path = tuple(q_out_nodes)

            if (best_intent_cost is None) or (final_cost < best_intent_cost):
                # Overwrite best cost
                intent_symbols_and_costs[q_intent] = [
                    FuzzyResult(
                        intent_name=q_intent, node_path=final_path, cost=final_cost
                    )
                ]
            elif final_cost == best_intent_cost:
                # Add to existing list
                intent_symbols_and_costs[q_intent].append(
                    (
                        FuzzyResult(
                            intent_name=q_intent, node_path=final_path, cost=final_cost
                        )
                    )
                )

            if final_cost < best_cost:
                # Update best cost so far
                best_cost = final_cost

        if q_cost > best_cost:
            # Can't get any better
            continue

        # Process child edges
        for next_node, edge_data in graph[q_node].items():
            in_label = edge_data.get("ilabel", "")
            out_label = edge_data.get("olabel", "")
            next_in_tokens = list(q_in_tokens)
            next_out_nodes = list(q_out_nodes)
            next_out_count = q_out_count
            next_cost = q_cost
            next_intent = q_intent

            if out_label:
                if out_label.startswith("__label__"):
                    next_intent = out_label[9:]
                elif not out_label.startswith("__"):
                    next_out_count += 1

            if in_label:
                if next_in_tokens:
                    if in_label == next_in_tokens[0]:
                        # Consume matching token
                        next_in_tokens.pop(0)
                    elif in_label in stop_words:
                        # Skip stop word (graph)
                        next_cost += stop_word_cost
                    elif next_in_tokens[0] in stop_words:
                        # Skip stop word (input)
                        next_cost += stop_word_cost
                    else:
                        # Mismatched token
                        next_cost += mismatched_word_cost
                elif in_label in stop_words:
                    # Skip stop word (graph)
                    next_cost += stop_word_cost
                else:
                    # No matching token
                    next_cost += extra_word_cost

            # Extend current path
            next_out_nodes.append(q_node)

            node_queue.append(
                (
                    next_node,
                    next_in_tokens,
                    next_out_nodes,
                    next_out_count,
                    next_cost,
                    next_intent,
                )
            )

    return intent_symbols_and_costs


def best_fuzzy_cost(
    intent_symbols_and_costs: typing.Dict[str, typing.List[FuzzyResult]]
) -> typing.List[FuzzyResult]:
    if not intent_symbols_and_costs:
        return []

    best_cost: typing.Optional[float] = None
    best_results: typing.List[FuzzyResult] = []

    # Find all results with the lowest cost
    for fuzzy_results in intent_symbols_and_costs.values():
        if not fuzzy_results:
            continue

        # All results for a given intent should have the same cost
        if best_cost is None:
            # First result
            best_cost = fuzzy_results[0].cost
            best_results = list(fuzzy_results)
        elif fuzzy_results[0].cost < best_cost:
            # Overwrite
            best_results = list(fuzzy_results)
        elif fuzzy_results[0].cost == best_cost:
            # Add to list
            best_results.extend(fuzzy_results)

    return best_results


# -----------------------------------------------------------------------------


def path_to_recognition(
    node_path: typing.Iterable[int],
    graph: nx.DiGraph,
    cost: typing.Optional[float] = None,
) -> typing.Tuple[RecognitionResult, typing.Optional[Recognition]]:
    """Transform node path in graph to an intent recognition object."""
    if not node_path:
        # Empty path indicates failure
        return RecognitionResult.FAILURE, None

    node_attrs = graph.nodes(data=True)
    recognition = Recognition(intent=Intent(""), confidence=1.0)

    # Text index for substituted and raw text
    sub_index = 0
    raw_index = 0

    # Named entities
    entity_stack: typing.List[Entity] = []

    # Handle first node
    node_path_iter = iter(node_path)
    last_node = next(node_path_iter)
    word = node_attrs[last_node].get("word", "")
    if word:
        recognition.raw_tokens.append(word)
        raw_index += len(word)

    # Follow path
    for next_node in node_path_iter:
        # Get raw text
        word = node_attrs[next_node].get("word", "")
        if word:
            recognition.raw_tokens.append(word)
            raw_index += len(word) + 1

            if entity_stack:
                last_entity = entity_stack[-1]
                last_entity.raw_tokens.append(word)

        # Get ilabel/olabel
        edge_data = graph[last_node][next_node]
        ilabel = edge_data.get("ilabel", "")
        olabel = edge_data.get("olabel", "")

        if olabel:
            if olabel.startswith("__label__"):
                # Intent
                recognition.intent.name = olabel[9:]
            elif olabel.startswith("__begin__"):
                # Begin entity
                entity_name = olabel[9:]
                entity_stack.append(
                    Entity(
                        entity=entity_name,
                        value="",
                        start=sub_index,
                        raw_start=raw_index,
                    )
                )
            elif olabel.startswith("__end__"):
                # End entity
                assert entity_stack, "Found __end__ without a __begin__"
                last_entity = entity_stack.pop()
                expected_name = olabel[7:]
                assert last_entity.entity == expected_name, "Mismatched entity name"

                # Assign end indexes
                last_entity.end = sub_index - 1
                last_entity.raw_end = raw_index - 1

                # Create values
                last_entity.value = " ".join(last_entity.tokens)
                last_entity.raw_value = " ".join(last_entity.raw_tokens)

                # Add to recognition
                recognition.intent.entities.append(last_entity)
            elif entity_stack:
                # Add to most recent named entity
                last_entity = entity_stack[-1]
                last_entity.tokens.append(olabel)

                recognition.tokens.append(olabel)
                sub_index += len(olabel) + 1
            else:
                # Substituted text
                recognition.tokens.append(olabel)
                sub_index += len(olabel) + 1

        last_node = next_node

    # Create text fields
    recognition.text = " ".join(recognition.tokens)
    recognition.raw_text = " ".join(recognition.raw_tokens)

    if cost and cost > 0:
        # Set fuzzy confidence
        recognition.confidence = 1 - (cost / len(recognition.raw_tokens))

    return RecognitionResult.SUCCESS, recognition
