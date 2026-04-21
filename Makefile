# Makefile — education-of-humanity replication repo
#
# make setup   — create venv + install dependencies
# make verify  — check every paper claim against data (~2 sec)
# make scripts — rebuild all checkin JSONs from source data

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip
PAPER_TEX = paper/education_of_humanity.tex
VERIFY_STAMP = checkin/.verified

.PHONY: all setup verify scripts clean

all: verify

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $@

verify: setup $(VERIFY_STAMP)

$(VERIFY_STAMP): checkin/*.json scripts/verify_humanity.py $(PAPER_TEX)
	$(PYTHON) scripts/verify_humanity.py --fast
	@touch $@

scripts: setup
	cd scripts && $(MAKE) PYTHON=$(abspath $(PYTHON))

clean:
	rm -rf $(VENV) $(VERIFY_STAMP)
