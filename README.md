# Rhasspy Natural Language Understanding

![Travis CI build status](https://travis-ci.com/synesthesiam/rhasspy-nlu.svg?branch=master)

Library for parsing Rhasspy sentence templates, doing intent recognition, and generating ARPA language models.

## Parsing Sentence Templates

Rhasspy voice commands are stored in text files formatted like this:

```ini
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

The result is a [directed graph](https://networx.github.io/documentation/networkx-2.3/reference/classes/digraph.html) whose states are words and edges are input/output labels.

You can pass an `intent_filter` function to `parse_ini` to return `True` for only the intent names you want to parse.
Additionally, a function can be provided for the `sentence_transform` argument that each sentence will be passed through (e.g., to lower case).

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
* Rules
    * `rule_name = rule body` can be referenced as `<rule_name>`
* Slots
    * `$slot` will be replaced by a list of sentences in the `replacements` argument of `intents_to_graph`
    
#### Rules

Named rules can be added to your template file using the syntax:

```ini
rule_name = rule body
```

and then reference using `<rule_name>`. The body of a rule is a regular sentence, which may itself contain references to other rules.

You can refrence rules from different intents by prefixing the rule name with the intent name and a dot:

```ini
[Intent1]
rule = a test
this is <rule>

[Intent2]
rule = this is
<rule> <Intent1.rule>
```

In the example above, `Intent2` uses its local `<rule>` as well as the `<rule>` from `Intent1`.

#### Slots

Slot names are prefixed with a dollar sign (`$`). When calling `intents_to_graph`, the `replacements` argument is a dictionary whose keys are slot names (with `$`) and whose values are lists of (parsed) `Sentence` objects. Each `$slot` will be replaced by the corresponding list of sentences, which may contain optional words, tags, rules, and other slots.

For example:

```python
import rhasspynlu

# Load and parse
intents = rhasspynlu.parse_ini(
"""
[SetColor]
set color to $color
"""
)

graph = rhasspynlu.intents_to_graph(
    intents, replacements = {
        "$color": [rhasspynlu.Sentence.parse("red | green | blue")]
    }
)
```

will replace `$color` with "red", "green", or "blue".

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

rhasspynlu.recognize("turn on living room lamp", graph)
```

will return a list of `Recognition` objects like:

```
[
    Recognition(
        intent=Intent(name='LightOn', confidence=1.0),
        entities=[
            Entity(
                entity='name',
                value='living room lamp',
                raw_value='living room lamp',
                start=8,
                raw_start=8,
                end=24,
                raw_end=24,
                tokens=['living', 'room', 'lamp'],
                raw_tokens=['living', 'room', 'lamp']
            )
        ],
        text='turn on living room lamp',
        raw_text='turn on living room lamp',
        recognize_seconds=0.00010710899914556649,
        tokens=['turn', 'on', 'living', 'room', 'lamp'],
        raw_tokens=['turn', 'on', 'living', 'room', 'lamp']
    )
]

```

An empty list means that recognition has failed. You can easily convert `Recognition` objects to JSON:

```python
...

import json

recognitions = rhasspynlu.recognize("turn on living room lamp", graph)
if recognitions:
    recognition_dict = recognitions[0].asdict()
    print(json.dumps(recognition_dict))
```

You can also pass an `intent_filter` function to `recognize` to return `True` only for intent names you want to include in the search.

#### Tokens

If your sentence is tokenized by something other than whitespace, pass the list of tokens into `recognize` instead of a string.

#### Recognition Fields

The `rhasspynlu.Recognition` object has the following fields:

* `intent` - a `rhasspynlu.Intent` instance
    * `name` - name of recognized intent
    * `confidence` - number for 0-1, 1 being sure
* `text` - substituted input text
* `raw_text` - input text
* `entities` - list of `rhasspynlu.Entity` objects
    * `entity` - name of recognized entity ("name" in `(input:output){name}`)
    * `value` - substituted value of recognized entity ("output" in `(input:output){name}`)
    * `tokens` - list of words in `value`
    * `start` - start index of `value` in `text`
    * `end` - end index of `value` in `text` (exclusive)
    * `raw_value` - value of recognized entity ("input" in `(input:output){name}`)
    * `raw_tokens` - list of words in `raw_value`
    * `raw_start` - start index of `raw_value` in `raw_text`
    * `raw_end` - end index of `raw_value` in `raw_text` (exclusive)
* `recognize_seconds` - seconds taken for `recognize`

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

Strict recognition also supports `stop_words` for a little added flexibility. If recognition without `stop_words` fails, a second attempt will be made using `stop_words`.

### Converters

Value conversions can be applied during recognition, such as converting the string "10" to the integer 10. Following a word, sequence, or tag name with "!converter" will run "converter" on the string value during `recognize`:

```python
import rhasspynlu

# Load and parse
intents = rhasspynlu.parse_ini(
"""
[SetBrightness]
set brightness to (one: hundred:100)!int
"""
)

graph = rhasspynlu.intents_to_graph(intents)

recognitions = rhasspynlu.recognize("set brightness to one hundred", graph)
assert recognitions[0].tokens[-1] == 100
```

Converters can be applied to tags/entities as well:

```python
import rhasspynlu

# Load and parse
intents = rhasspynlu.parse_ini(
"""
[SetBrightness]
set brightness to (one:1 | two:2){value!int}
"""
)

graph = rhasspynlu.intents_to_graph(intents)

recognitions = rhasspynlu.recognize("set brightness to two", graph)
assert recognitions[0].tokens[-1] == 2
```

The following default converters are available in `rhasspynlu`:

* int - convert to integer
* float - convert to real
* bool - convert to boolean
* lower - lower-case
* upper - upper-case

You may override these converters by passing a dictionary to the `converters` argument of `recognize`. To supply additional converters (instead of overriding), use `extra_converters`:

```python
import rhasspynlu

# Load and parse
intents = rhasspynlu.parse_ini(
"""
[SetBrightness]
set brightness to (one:1 | two:2){value!myconverter}
"""
)

graph = rhasspynlu.intents_to_graph(intents)

recognitions = rhasspynlu.recognize(
    "set brightness to two",
    graph,
    extra_converters={
        "myconverter": lambda *values: [int(v)**2 for v in values]
    }
)
assert recognitions[0].tokens[-1] == 4
```

Lastly, you can chain converters together with multiple "!":

```python
import rhasspynlu

# Load and parse
intents = rhasspynlu.parse_ini(
"""
[SetBrightness]
set brightness to (one:1 | two:2){value!int!cube}
"""
)

graph = rhasspynlu.intents_to_graph(intents)

recognitions = rhasspynlu.recognize(
    "set brightness to two",
    graph,
    extra_converters={
        "cube": lambda *values: [v**3 for v in values]
    }
)
assert recognitions[0].tokens[-1] == 8
```

## ARPA Language Models

You can compute [ngram counts](https://en.wikipedia.org/wiki/N-gram) from a `rhasspynlu` graph, useful for generating [ARPA language models](https://cmusphinx.github.io/wiki/arpaformat/). These models can be used by speech recognition systems, such as [Pocketsphinx](https://github.com/cmusphinx/pocketsphinx), [Kaldi](https://kaldi-asr.org), and [Julius](https://github.com/julius-speech/julius).

```python
import rhasspynlu

# Load and parse
intents = rhasspynlu.parse_ini(
"""
[SetColor]
set light to (red | green | blue)
"""
)

graph = rhasspynlu.intents_to_graph(intents)
counts = rhasspynlu.get_intent_ngram_counts(
    graph,
    pad_start="<s>",
    pad_end="</s>",
    order=3
)

# Print counts by intent
for intent_name in counts:
    print(intent_name)
    for ngram, count in counts[intent_name].items():
        print(ngram, count)
        
    print("")
```

will print something like:

```
SetColor
('<s>',) 3
('set',) 3
('<s>', 'set') 3
('light',) 3
('set', 'light') 3
('<s>', 'set', 'light') 3
('to',) 3
('light', 'to') 3
('set', 'light', 'to') 3
('red',) 1
('to', 'red') 1
('light', 'to', 'red') 1
('green',) 1
('to', 'green') 1
('light', 'to', 'green') 1
('blue',) 1
('to', 'blue') 1
('light', 'to', 'blue') 1
('</s>',) 3
('red', '</s>') 1
('green', '</s>') 1
('blue', '</s>') 1
('to', 'red', '</s>') 1
('to', 'green', '</s>') 1
('to', 'blue', '</s>') 1

```

### Opengrm

If you have the [Opengrm](http://www.opengrm.org/twiki/bin/view/GRM/NGramLibrary) command-line tools in your `PATH`, you can use `rhasspynlu` to generate language models in the [ARPA format](https://cmusphinx.github.io/wiki/arpaformat/). 

The `graph_to_fst` and `fst_to_arpa` functions are used to convert between formats. Calling `fst_to_arpa` requires the following binaries to be present in your `PATH`:

* `fstcompile` (from [OpenFST](http://www.openfst.org))
* `ngramcount`
* `ngrammake`
* `ngrammerge`
* `ngramprint`
* `ngramread`

Example:

```python
# Convert to FST
graph_fst = rhasspynlu.graph_to_fst(graph)

# Write FST and symbol text files
graph_fst.write("my_fst.txt", "input_symbols.txt", "output_symbols.txt")

# Compile and convert to ARPA language model
rhasspynlu.fst_to_arpa(
    "my_fst.txt",
    "input_symbols.txt", 
    "output_symbols.txt",
    "my_arpa.lm"
)
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
rhasspynlu.fst_to_arpa(
    "my_fst.txt",
    "input_symbols.txt",
    "output_symbols.txt",
    "my_arpa.lm",
    base_fst_weight=("existing_arpa.fst", 0.05)
)
```

## Command Line Usage

The `rhasspynlu` module can be run directly to convert `sentences.ini` files into JSON graphs or FST text files:

```bash
python3 -m rhasspynlu sentences.ini > graph.json
```

You can pass multiple `.ini` files as arguments, and they will be combined. Adding a `--fst` argument will write out FST text files instead:

```
python3 -m rhasspynlu sentences.ini --fst
```

This will output three files in the current directory:

* `fst.txt` - finite state transducer as text
* `fst.isymbols.txt` - input symbols
* `fst.osymbols.txt` - output symbols

These file names can be changed with the `--fst-text`, `--fst-isymbols`, and `--fst-osymbols` arguments, respectively.

Compile to a binary FST using `fstcompile` (from [OpenFST](http://www.openfst.org)) with:

```bash
fstcompile \
    --isymbols=fst.isymbols.txt \
    --osymbols=fst.osymbols.txt \
    --keep_isymbols=1 \
    --keep_osymbols=1 \
    fst.txt \
    out.fst
```
