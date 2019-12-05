"""Utilities to convert JSGF sentences to directed graphs."""
from collections import defaultdict
from enum import Enum
import typing

import attr
import networkx as nx

from .jsgf import (
    Sentence,
    Rule,
    RuleReference,
    SlotReference,
    Word,
    Sequence,
    SequenceType,
    Expression,
    Taggable,
    Substitutable,
)

# -----------------------------------------------------------------------------


def expression_to_graph(
    expression: Expression,
    graph: nx.DiGraph,
    source_state: int,
    replacements: typing.Dict[str, typing.Iterable[Sentence]] = None,
    empty_substitution: bool = False,
    grammar_name: typing.Optional[str] = None,
) -> int:
    """Insert JSGF expression into a graph. Return final state."""
    replacements: typing.Dict[str, typing.Iterable[Sentence]] = replacements or {}

    # Handle sequence substitution
    if isinstance(expression, Substitutable) and expression.substitution:
        # Ensure everything downstream outputs nothing
        empty_substitution = True

    # Handle tag begin
    if isinstance(expression, Taggable) and expression.tag:
        # Begin tag
        next_state = len(graph)
        tag = expression.tag.tag_text
        olabel = f"__begin__{tag}"
        label = f":{olabel}"
        graph.add_edge(source_state, next_state, ilabel="", olabel=olabel, label=label)
        source_state = next_state

        if expression.tag.substitution:
            # Ensure everything downstream outputs nothing
            empty_substitution = True

    if isinstance(expression, Sequence):
        # Group, optional, or alternative
        seq: Sequence = expression
        if seq.type == SequenceType.ALTERNATIVE:
            # Optional or alternative
            final_states = []
            for item in seq.items:
                # Branch alternatives from source state
                next_state = expression_to_graph(
                    item,
                    graph,
                    source_state,
                    replacements=replacements,
                    empty_substitution=empty_substitution,
                    grammar_name=grammar_name,
                )
                final_states.append(next_state)

            # Connect all paths to final state
            next_state = len(graph)
            for final_state in final_states:
                graph.add_edge(final_state, next_state, ilabel="", olabel="", label="")

            source_state = next_state
        else:
            # Group
            next_state = source_state
            for item in seq.items:
                # Create sequence of states
                next_state = expression_to_graph(
                    item,
                    graph,
                    next_state,
                    replacements=replacements,
                    empty_substitution=empty_substitution,
                    grammar_name=grammar_name,
                )

            source_state = next_state
    elif isinstance(expression, Word):
        # State for single word
        word: Word = expression
        next_state = len(graph)
        graph.add_node(next_state, word=word.text)
        ilabel = word.text
        olabel = word.substitution or word.text

        if empty_substitution:
            olabel = ""

        label = f"{ilabel}:{olabel}"
        graph.add_edge(
            source_state, next_state, ilabel=ilabel, olabel=olabel, label=label
        )
        source_state = next_state
    elif isinstance(expression, RuleReference):
        # Reference to a local or remote rule
        rule_ref: RuleReference = expression
        if rule_ref.grammar_name:
            # Fully resolved rule name
            rule_name = f"{rule_ref.grammar_name}.{rule_ref.rule_name}"
        elif grammar_name:
            # Local rule
            rule_name = f"{grammar_name}.{rule_ref.rule_name}"
        else:
            # Unresolved rule name
            rule_name = rule_ref.rule_name

        # Surround with <>
        rule_name = f"<{rule_name}>"
        rule_replacements = replacements.get(rule_name)
        assert rule_replacements, f"Missing rule {rule_name}"

        rule_body = rule_replacements[0]
        assert isinstance(rule_body, Sentence), f"Invalid rule {rule_name}: {rule_body}"
        source_state = expression_to_graph(
            rule_body,
            graph,
            source_state,
            replacements=replacements,
            empty_substitution=empty_substitution,
            grammar_name=grammar_name,
        )
    elif isinstance(expression, SlotReference):
        # Reference to slot values
        slot_ref: SlotReference = expression

        # Prefix with $
        slot_name = "$" + slot_ref.slot_name
        slot_values = replacements.get(slot_name)
        assert slot_values, f"Missing slot {slot_values}"

        # Interpret as alternative
        slot_seq = Sequence(type=SequenceType.ALTERNATIVE, items=slot_values)
        source_state = expression_to_graph(
            slot_seq,
            graph,
            source_state,
            replacements=replacements,
            empty_substitution=(empty_substitution or slot_ref.substitution),
            grammar_name=grammar_name,
        )

    # Handle sequence substitution
    if isinstance(expression, Substitutable) and expression.substitution:
        # Output substituted word
        next_state = len(graph)
        olabel = expression.substitution
        label = f":{olabel}"
        graph.add_edge(source_state, next_state, ilabel="", olabel=olabel, label=label)
        source_state = next_state

    # Handle tag end
    if isinstance(expression, Taggable) and expression.tag:
        # Handle tag substitution
        if expression.tag.substitution:
            # Output substituted word
            next_state = len(graph)
            olabel = expression.tag.substitution
            label = f":{olabel}"
            graph.add_edge(
                source_state, next_state, ilabel="", olabel=olabel, label=label
            )
            source_state = next_state

        # End tag
        next_state = len(graph)
        tag = expression.tag.tag_text
        olabel = f"__end__{tag}"
        label = f":{olabel}"
        graph.add_edge(source_state, next_state, ilabel="", olabel=olabel, label=label)
        source_state = next_state

    return source_state


# -----------------------------------------------------------------------------


def intents_to_graph(
    intents: typing.Dict[str, typing.List[typing.Union[Sentence, Rule]]],
    replacements: typing.Dict[str, typing.Iterable[Sentence]] = None,
) -> nx.DiGraph:
    """Convert sentences/rules grouped by intent into a directed graph."""
    # Slots or rules
    replacements: typing.Dict[str, typing.Iterable[Sentence]] = replacements or {}

    # Strip rules from intents
    sentences: typing.Dict[str, typing.List[Sentence]] = defaultdict(list)
    for intent_name, intent_items in intents.items():
        for item in intent_items:
            if isinstance(item, Rule):
                # Rule
                rule_name = item.rule_name
                if "." not in rule_name:
                    rule_name = f"{intent_name}.{rule_name}"

                # Surround with <>
                rule_name = f"<{rule_name}>"
                replacements[rule_name] = [item.rule_body]
            else:
                # Sentence
                sentences[intent_name].append(item)

    # Create initial graph
    graph: nx.DiGraph = nx.DiGraph()
    root_state: int = 0
    graph.add_node(root_state, start=True)
    final_states: typing.List[int] = []

    for intent_name, intent_sentences in sentences.items():
        # Branch off for each intent from start state
        intent_state = len(graph)
        olabel = f"__label__{intent_name}"
        label = f":{olabel}"
        graph.add_edge(root_state, intent_state, ilabel="", olabel=olabel, label=label)

        for sentence in intent_sentences:
            # Insert all sentences for this intent
            next_state = expression_to_graph(
                sentence,
                graph,
                intent_state,
                replacements=replacements,
                grammar_name=intent_name,
            )
            final_states.append(next_state)

    # Create final state and join all sentences to it
    final_state = len(graph)
    graph.add_node(final_state, final=True)

    for next_state in final_states:
        graph.add_edge(next_state, final_state, ilabel="", olabel="", label="")

    return graph
