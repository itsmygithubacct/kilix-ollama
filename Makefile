PYTHON ?= python3
INSTALL ?= install
PREFIX ?= /usr/local
DESTDIR ?=

.DEFAULT_GOAL := check

check: test
	$(PYTHON) -m compileall -q src tests kilix-ollama

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

install:
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/bin
	$(INSTALL) -m 0755 kilix-ollama $(DESTDIR)$(PREFIX)/bin/kilix-ollama
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/lib/kilix-ollama/kilix_ollama
	$(INSTALL) -m 0644 src/kilix_ollama/*.py \
		$(DESTDIR)$(PREFIX)/lib/kilix-ollama/kilix_ollama/

.PHONY: check install test
