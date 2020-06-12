all: check test coverage lint

lint:
	@flake8 imrsv
	@mypy --strict imrsv

.IGNORE:
check:
	@flake8 --select=F imrsv
	@mypy imrsv

test:
	@pytest -qq -- tests

last:
	@pytest -qq --lf -- tests

coverage:
	@coverage html

cloc:
	@cloc imrsv tests

tags:
	@ctags -R imrsv tests

clean:
	@rm -rf .coverage htmlcov .mypy_cache tags

.PHONY: all lint check test coverage clean tags
