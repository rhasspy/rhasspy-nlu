SOURCE = rhasspynlu
PYTHON_FILES = $(SOURCE)/*.py tests/*.py *.py
PIP_INSTALL ?= install

.PHONY: reformat check test dist venv

reformat:
	black .
	isort $(PYTHON_FILES)

check:
	flake8 $(PYTHON_FILES)
	pylint $(PYTHON_FILES)
	mypy $(PYTHON_FILES)
	black --check .
	isort --check-only $(PYTHON_FILES)
	yamllint .
	pip list --outdated

test:
	coverage run --source=$(SOURCE) -m unittest
	coverage report -m
	coverage xml

venv:
	scripts/create-venv.sh

dist:
	python3 setup.py sdist
