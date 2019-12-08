# Rhasspy Natural Language Understanding

Library for parsing Rhasspy sentence templates, doing intent recognition, and generating ARPA language models.

## Parsing Sentence Templates

Rhasspy voice commands are stored in text files formatted like this:

```
[Intent1]
this is a sentence
this is another sentence

[Intent2]
a sentence in a different intent
```

You can parse these into a structured representation with `rhasspynlu.parse_ini` and then convert them to a graph using `rhasspynlu.intents_to_graph`:

```python
import rhasspynlu

# Load and parse
intents = rhasspynlu.parse_ini(
"""
[LightOn]
turn on [the] (living room lamp | kitchen light){name}
"""
)

graph = rhasspynlu.intents_to_graph(intents)
```

You can pass an `intent_filter` function to `parse_ini` to return `True` for only the intents you want to include. 

### Template Syntax

Sentence templates are based on the [JSGF](https://www.w3.org/TR/jsgf/) standard. The following constructs are available:

* Optional words
    * `this is [a] test` - the word "a" may or may not be present
* Alternatives
    * `set color to (red | green | blue)` - either "red", "green", or "blue" is possible
* Tags
    * `turn on the [den | playroom]{location} light` - named entity `location` will be either "den" or "playroom"
* Substitutions
    * `make ten:10 coffees` - output will be "make 10 coffees"
    * `turn off the: (television | tele):tv` - output will be "turn off tv"
    * `set brightness to (medium | half){brightness:50}` - named entity `brightness` will be "50"

## Intent Recognition

After converting your sentence templates to a graph, you can recognize sentences. Assuming you have a `.ini` file like this:

```
[LightOn]
turn on [the] (living room lamp | kitchen light){name}
```

You can recognize sentences with:

```python
from pathlib import Path
import rhasspynlu

# Load and parse
intents = rhasspynlu.parse_ini(Path("sentences.ini"))
graph = rhasspynlu.intents_to_graph(intents)

print(rhasspynlu.recognize("turn on living room lamp", graph))
```

The `recognize` function returns a list of `Recognition` objects (or nothing if recognition fails). You can convert these to JSON:

```python
import json

recognitions = rhasspynlu.recognize("turn on living room lamp", graph)
if recognitions:
    recognition_dict = recognitions[0].asdict()
    print(json.dumps(recognition_dict))
```

You can also pass an `intent_filter` function to `recognize` to return `True` for only the intents you want to include.

### Stop Words

You can pass a set of `stop_words` to `recognize`:

```python
rhasspynlu.recognize("turn on that living room lamp", graph, stop_words=set(["that"]))
```

Stop words in the input sentence will be skipped over if they don't match the graph.

### Strict Recognition

For faster, but less flexible recognition, set `fuzzy` to `False`:

```python
rhasspynlu.recognize("turn on the living room lamp", graph, fuzzy=False)
```

This is at least twice as fast, but will fail if the sentence is not precisely present in the graph.

Strict recognition also supports `stop_words` for a little added flexibility.

## ARPA Language Models

If you have the [Opengrm](http://www.opengrm.org/twiki/bin/view/GRM/NGramLibrary) command-line tools in your `PATH`, you can use `rhasspynlu` to generate language models in the [ARPA format](https://cmusphinx.github.io/wiki/arpaformat/). 
These models can be used by speech recognition systems, such as [Pocketsphinx](https://github.com/cmusphinx/pocketsphinx), [Kaldi](https://kaldi-asr.org), and [Julius](https://github.com/julius-speech/julius).

The `graph_to_fst` and `fst_to_arpa` functions are used to convert between formats. Calling `fst_to_arpa` requires the following binaries to be present in your `PATH`:

* `fstcompile` (from [OpenFST](http://www.openfst.org))
* `ngramcount`
* `ngrammake`
* `ngrammerge`
* `ngramprint`
* `ngramread`

Example:

```python
...

# Convert to FST
graph_fst = rhasspynlu.graph_to_fst(graph)

# Write FST and symbol text files
graph_fst.write("my_fst.txt", "input_symbols.txt", "output_symbols.txt")

# Compile and convert to ARPA language model
rhasspynlu.fst_to_arpa("my_fst.txt", "input_symbols.txt", "output_symbols.txt", "my_arpa.lm")
```

You can now use `my_arpa.lm` in any speech recognizer that accepts ARPA-formatted language models.

### Language Model Mixing

If you have an existing language model that you'd like to mix with Rhasspy voice commands, you will first need to convert it to an FST:

```python
rhasspynlu.fst_to_arpa("existing_arpa.lm", "existing_arpa.fst")
```

Now when you call `fst_to_arpa`, make sure to provide the `base_fst_weight` argument. This is a tuple with the path to your existing ARPA FST and a mixture weight between 0 and 1. A weight of 0.05 means that the base language model will receive 5% of the overall probability mass in the language model. The rest of the mass will be given to your custom voice commands.

Example:

```python
...
rhasspynlu.fst_to_arpa("my_fst.txt", "input_symbols.txt", "output_symbols.txt", "my_arpa.lm", base_fst_weight=("existing_arpa.fst", 0.05))
```
