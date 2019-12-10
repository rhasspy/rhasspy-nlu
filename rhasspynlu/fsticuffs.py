"""Recognition functions for sentences using JSGF graphs."""
from collections import defaultdict
import typing

import attr
import networkx as nx

from .intent import Entity, Intent, Recognition, RecognitionResult

# -----------------------------------------------------------------------------


def recognize(
    tokens: typing.Union[str, typing.List[str]],
    graph: nx.DiGraph,
    fuzzy: bool = True,
    stop_words: typing.Optional[typing.Set[str]] = None,
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
    **search_args
) -> typing.List[Recognition]:
    """Recognize one or more intents from tokens or a sentence."""
    if isinstance(tokens, str):
        # Assume whitespace separation
        tokens = tokens.split()

    if fuzzy:
        # Fuzzy recognition
        best_fuzzy = best_fuzzy_cost(
            paths_fuzzy(
                tokens,
                graph,
                stop_words=stop_words,
                intent_filter=intent_filter,
                **search_args
            )
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

            return recognitions
    else:
        # Strict recognition
        paths = list(
            paths_strict(tokens, graph, intent_filter=intent_filter, **search_args)
        )
        if (not paths) and stop_words:
            # Try again by excluding stop words
            tokens = [t for t in tokens if t not in stop_words]
            paths = list(
                paths_strict(
                    tokens,
                    graph,
                    exclude_tokens=stop_words,
                    intent_filter=intent_filter,
                    **search_args
                )
            )

        recognitions = []
        for path in paths:
            result, recognition = path_to_recognition(path, graph)
            if result == RecognitionResult.SUCCESS:
                recognitions.append(recognition)

        return recognitions

    # No results
    return []


# -----------------------------------------------------------------------------


def paths_strict(
    tokens: typing.List[str],
    graph: nx.DiGraph,
    exclude_tokens: typing.Optional[typing.Set[str]] = None,
    max_paths: typing.Optional[int] = None,
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
) -> typing.Iterable[typing.List[int]]:
    """Match a single path from the graph exactly if possible."""
    if not tokens:
        return []

    intent_filter = intent_filter or (lambda x: True)

    # node -> attrs
    n_data = graph.nodes(data=True)

    # start state
    start_node: int = [n for n, data in n_data if data.get("start", False)][0]

    # Number of matching paths found
    paths_found: int = 0

    # Do breadth-first search
    node_queue = [(start_node, [], tokens)]
    while node_queue:
        current_node, current_path, current_tokens = node_queue.pop(0)
        is_final = n_data[current_node].get("final", False)
        if is_final and (not current_tokens):
            # Reached final state
            paths_found += 1
            yield current_path

            if max_paths and (paths_found >= max_paths):
                break

        for next_node, edge_data in graph[current_node].items():
            next_path = list(current_path)
            next_tokens = list(current_tokens)

            ilabel = edge_data.get("ilabel", "")
            olabel = edge_data.get("olabel", "")

            if olabel.startswith("__label__"):
                intent_name = olabel[9:]
                if not intent_filter(intent_name):
                    # Skip intent
                    continue

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

    # No results
    return []


# -----------------------------------------------------------------------------


@attr.s
class FuzzyResult:
    """Single path for fuzzy recognition."""

    intent_name: str = attr.ib()
    node_path: typing.Iterable[int] = attr.ib()
    cost: float = attr.ib()


@attr.s
class FuzzyCostInput:
    """Input to fuzzy cost function."""

    ilabel: str = attr.ib()
    tokens: typing.Deque[str] = attr.ib()
    stop_words: typing.Set[str] = attr.ib()


@attr.s
class FuzzyCostOutput:
    """Output from fuzzy cost function."""

    cost: float = attr.ib()
    continue_search: bool = attr.ib(default=True)


def default_fuzzy_cost(cost_input: FuzzyCostInput) -> FuzzyCostOutput:
    """Increases cost when input tokens fail to match graph. Marginal cost for stop words."""
    ilabel: str = cost_input.ilabel
    cost: float = 0.0
    tokens: typing.Deque[str] = cost_input.tokens
    stop_words: typing.Set[str] = cost_input.stop_words

    if ilabel:
        while tokens and (ilabel != tokens[0]):
            bad_token = tokens.pop(0)
            if bad_token in stop_words:
                # Marginal cost to ensure paths matching stop words are preferred
                cost += 0.1
            else:
                # Mismatched token
                cost += 1

        if tokens and (ilabel == tokens[0]):
            # Consume matching token
            tokens.pop(0)
        else:
            # No matching token
            return FuzzyCostOutput(cost=cost, continue_search=False)

    return FuzzyCostOutput(cost=cost)


def paths_fuzzy(
    tokens: typing.List[str],
    graph: nx.DiGraph,
    stop_words: typing.Optional[typing.Set[str]] = None,
    cost_function: typing.Optional[
        typing.Callable[[FuzzyCostInput], FuzzyCostOutput]
    ] = None,
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
) -> typing.Dict[str, typing.List[FuzzyResult]]:
    """Do less strict matching using a cost function and optional stop words."""
    if not tokens:
        return []

    intent_filter = intent_filter or (lambda x: True)
    cost_function = cost_function or default_fuzzy_cost
    stop_words: typing.Set[str] = stop_words or set()

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
    node_queue: typing.List[
        typing.Tuple[
            int, typing.List[str], typing.List[int], int, float, typing.Optional[str]
        ]
    ] = [(start_node, tokens, [], 0, 0.0, None)]

    # BFS it up
    while node_queue:
        q_node, q_in_tokens, q_out_nodes, q_out_count, q_cost, q_intent = node_queue.pop(
            0
        )
        is_final: bool = n_data[q_node].get("final", False)

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
                    if not intent_filter(next_intent):
                        # Skip intent
                        continue
                elif not out_label.startswith("__"):
                    next_out_count += 1

            cost_output = cost_function(
                FuzzyCostInput(
                    ilabel=in_label, tokens=next_in_tokens, stop_words=stop_words
                )
            )

            next_cost += cost_output.cost

            if not cost_output.continue_search:
                continue

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
    """Return fuzzy results with cost."""
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

        # Get output label
        edge_data = graph[last_node][next_node]
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
