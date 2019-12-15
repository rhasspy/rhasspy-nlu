.PHONY: check test dist venv

check:
	flake8 rhasspynlu/*.py rhasspynlu/test/*.py
	pylint rhasspynlu/*.py rhasspynlu/test/*.py

test:
	python3 -m unittest \
    rhasspynlu.test.jsgf_test \
    rhasspynlu.test.ini_jsgf_test \
    rhasspynlu.test.jsgf_graph_test \
    rhasspynlu.test.fsticuffs_test

venv:
	rm -rf .venv/
	python3 -m venv .venv
	.venv/bin/pip3 install wheel setuptools
	.venv/bin/pip3 install -r requirements_all.txt

dist:
	python3 setup.py sdist
