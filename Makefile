.PHONY: style types tests coverage

style:
	ruff format src tests
	ruff check --fix src tests

types:
	mypy --strict src

tests:
	python -m unittest

coverage:
	coverage run --source src/myks_gopro,tests -m unittest
	coverage html
