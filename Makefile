SHELL := /bin/bash

include .env

dev:
	uv run python -m app.main