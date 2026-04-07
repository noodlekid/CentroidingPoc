PYTHON ?= $(shell poetry env info --executable)
PYSPY ?= $(dir $(PYTHON))py-spy
SNAKEVIZ ?= $(dir $(PYTHON))snakeviz
PYINSTRUMENT ?= $(dir $(PYTHON))pyinstrument

PROFILE_MODULE := crowsnest.poc.eval.profile_centroiding
ARTIFACTS_DIR := artifacts

WIDTH ?= 256
HEIGHT ?= 192
NUM_STARS ?= 20
SEED ?= 42
ITERATIONS ?= 3000

CPROFILE_FILE ?= $(ARTIFACTS_DIR)/centroiding.prof
FLAMEGRAPH_FILE ?= $(ARTIFACTS_DIR)/flamegraph.html
SPEEDSCOPE_FILE ?= $(ARTIFACTS_DIR)/profile.speedscope.json

COMMON_PROFILE_ARGS := --iterations $(ITERATIONS) --width $(WIDTH) --height $(HEIGHT) --num-stars $(NUM_STARS) --seed $(SEED)

.PHONY: help profile-loop cprofile flamegraph pyspy-flamegraph speedscope snakeviz

help:
	@echo "Targets:"
	@echo "  make profile-loop   # run centroiding loop only"
	@echo "  make cprofile       # generate cProfile .prof file"
	@echo "  make flamegraph     # generate flamegraph-style HTML using pyinstrument"
	@echo "  make pyspy-flamegraph # generate py-spy SVG flamegraph (may fail on Python 3.14)"
	@echo "  make speedscope     # generate speedscope JSON using py-spy"
	@echo "  make snakeviz       # open snakeviz for generated .prof"
	@echo ""
	@echo "Override defaults (example):"
	@echo "  make flamegraph ITERATIONS=6000 WIDTH=800 HEIGHT=600 NUM_STARS=60"

$(ARTIFACTS_DIR):
	mkdir -p $(ARTIFACTS_DIR)

profile-loop:
	$(PYTHON) -m $(PROFILE_MODULE) $(COMMON_PROFILE_ARGS)

cprofile: | $(ARTIFACTS_DIR)
	$(PYTHON) -m $(PROFILE_MODULE) $(COMMON_PROFILE_ARGS) --cprofile-output $(CPROFILE_FILE)
	@echo "Saved: $(CPROFILE_FILE)"

flamegraph: | $(ARTIFACTS_DIR)
	$(PYINSTRUMENT) -r html -o $(FLAMEGRAPH_FILE) -m $(PROFILE_MODULE) $(COMMON_PROFILE_ARGS)
	@echo "Saved: $(FLAMEGRAPH_FILE)"

pyspy-flamegraph: | $(ARTIFACTS_DIR)
	$(PYSPY) record --format flamegraph -o $(ARTIFACTS_DIR)/flamegraph.svg -- $(PYTHON) -m $(PROFILE_MODULE) $(COMMON_PROFILE_ARGS)
	@echo "Saved: $(ARTIFACTS_DIR)/flamegraph.svg"

speedscope: | $(ARTIFACTS_DIR)
	$(PYSPY) record --format speedscope -o $(SPEEDSCOPE_FILE) -- $(PYTHON) -m $(PROFILE_MODULE) $(COMMON_PROFILE_ARGS)
	@echo "Saved: $(SPEEDSCOPE_FILE)"

snakeviz:
	$(SNAKEVIZ) $(CPROFILE_FILE)
