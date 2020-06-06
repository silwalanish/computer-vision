all: run

run:
	@pipenv run python src/main.py

lint:
	@black src

.PHONY = run lint
