docs:
	$(MAKE) -C docs html

package:
	python -m build .

test:
	python -m pytest

.PHONY: docs package test
