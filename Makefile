.PHONY: test

test:
	python3 -m unittest \
    rhasspynlu.test.jsgf_test \
    rhasspynlu.test.ini_jsgf_test \
    rhasspynlu.test.jsgf_graph_test \
    rhasspynlu.test.fsticuffs_test
