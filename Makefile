.PHONY: check test dist

check:
	flake8 rhasspynlu/*.py rhasspynlu/test/*.py
	pylint rhasspynlu/*.py rhasspynlu/test/*.py

test:
	python3 -m unittest \
    rhasspynlu.test.jsgf_test \
    rhasspynlu.test.ini_jsgf_test \
    rhasspynlu.test.jsgf_graph_test \
    rhasspynlu.test.fsticuffs_test

dist:
	python3 setup.py sdist
