.PHONY: check test dist venv

check:
	flake8 rhasspynlu/*.py test/*.py
	pylint rhasspynlu/*.py test/*.py
	mypy rhasspynlu/*.py test/*.py

test:
	python3 -m unittest \
    test.jsgf_test \
    test.ini_jsgf_test \
    test.jsgf_graph_test \
    test.fsticuffs_test

venv:
	rm -rf .venv/
	python3 -m venv .venv
	.venv/bin/pip3 install wheel setuptools
	.venv/bin/pip3 install -r requirements_all.txt

dist:
	python3 setup.py sdist
