.PHONY: check test dist venv

check:
	flake8 rhasspynlu/*.py tests/*.py
	mypy rhasspynlu/*.py tests/*.py
	pylint rhasspynlu/*.py tests/*.py

test:
	python3 -m unittest \
    tests.test_jsgf \
    tests.test_ini_jsgf \
    tests.test_jsgf_graph \
    tests.test_fsticuffs

venv:
	rm -rf .venv/
	python3 -m venv .venv
	.venv/bin/pip3 install wheel setuptools
	.venv/bin/pip3 install -r requirements_all.txt

dist:
	python3 setup.py sdist
