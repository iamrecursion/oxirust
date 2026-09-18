.PHONY: dev test clean

dev:
	maturin develop --release

test: dev
	pytest -v

clean:
	rm -rf target/wheels build/ *.egg-info __pycache__ .pytest_cache
