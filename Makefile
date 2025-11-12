# Makefile for SLICUTLET - simplifies build, test, and development workflow
#
# Common targets:
#   make build      - Build the project
#   make test       - Run all tests
#   make test-mb01  - Run MB01 tests only
#   make clean      - Clean build artifacts
#   make install    - Build and install to build-install/
#   make dev        - Build, install, and run tests

# Configuration
PYTHON := uv run
UV := uv

# Build directories
BUILD_DIR := build
INSTALL_DIR := $(PWD)/build-install
INSTALL_LIB := $(INSTALL_DIR)/usr/local/lib
INSTALL_PYLIB := $(INSTALL_DIR)/usr/local/lib/python3.13/site-packages

# Environment for running tests (must be inline for uv run)
ENV_VARS := DYLD_LIBRARY_PATH=$(INSTALL_LIB):$$DYLD_LIBRARY_PATH PYTHONPATH=$(INSTALL_PYLIB):$$PYTHONPATH
PYTEST := $(ENV_VARS) uv run pytest

# Test options
PYTEST_OPTS := -v
TEST_DIR := python/tests

.PHONY: help
help:
	@echo "SLICUTLET Makefile"
	@echo ""
	@echo "Setup targets:"
	@echo "  make setup         - Sync dependencies from pyproject.toml (uv sync)"
	@echo ""
	@echo "Build targets:"
	@echo "  make build         - Compile C library and Python extension"
	@echo "  make install       - Install to build-install/ directory"
	@echo "  make rebuild       - Clean and rebuild from scratch"
	@echo "  make configure     - Configure meson (auto-runs setup)"
	@echo ""
	@echo "Test targets:"
	@echo "  make test          - Run all tests"
	@echo "  make test-mb01     - Run MB01 family tests"
	@echo "  make test-ma01     - Run MA01 family tests"
	@echo "  make test-ma02     - Run MA02 family tests"
	@echo "  make test-ab       - Run AB family tests"
	@echo "  make test-mc       - Run MC family tests"
	@echo "  make test-one TEST=test_name  - Run specific test"
	@echo ""
	@echo "Development targets:"
	@echo "  make dev           - Build, install, and test (full cycle)"
	@echo "  make quick-test    - Test without rebuilding"
	@echo "  make format        - Format Python code with ruff"
	@echo "  make lint          - Check Python code with ruff"
	@echo "  make lint-fix      - Auto-fix ruff issues"
	@echo ""
	@echo "Package management:"
	@echo "  uv add <package>   - Add new package to pyproject.toml"
	@echo "  uv remove <package> - Remove package from pyproject.toml"
	@echo "  uv sync            - Sync installed packages with pyproject.toml"
	@echo ""
	@echo "Cleanup targets:"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make clean-all     - Remove build artifacts and venv"
	@echo "  make clean-editable - Remove editable install artifacts"

# Setup and dependency installation
.PHONY: setup
setup:
	$(UV) sync

# Build targets
.PHONY: build
build:
	$(UV) run meson compile -C $(BUILD_DIR)

.PHONY: configure
configure: setup
	$(UV) run meson setup $(BUILD_DIR) -Dpython=true

.PHONY: install
install: build
	$(UV) run meson install -C $(BUILD_DIR) --destdir="$(INSTALL_DIR)"
	@$(MAKE) clean-editable

.PHONY: rebuild
rebuild: clean configure install

.PHONY: dev
dev: install test

# Test targets
.PHONY: test
test:
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)/

.PHONY: test-mb01
test-mb01:
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)/test_mb01xx.py

.PHONY: test-ma01
test-ma01:
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)/test_ma01xx.py

.PHONY: test-ma02
test-ma02:
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)/test_ma02xx.py

.PHONY: test-ab
test-ab:
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)/test_ab*.py

.PHONY: test-mc
test-mc:
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)/test_mc*.py

.PHONY: test-one
test-one:
	@if [ -z "$(TEST)" ]; then \
		echo "Error: TEST variable not set. Usage: make test-one TEST=test_name"; \
		exit 1; \
	fi
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)/ -k "$(TEST)"

.PHONY: quick-test
quick-test:
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)/

# Code quality targets
.PHONY: format
format:
	$(UV) run ruff format python/

.PHONY: lint
lint:
	$(UV) run ruff check python/

.PHONY: lint-fix
lint-fix:
	$(UV) run ruff check --fix python/

# Cleanup targets
.PHONY: clean-editable
clean-editable:
	@echo "Cleaning editable install artifacts..."
	@rm -f .venv/lib/python*/site-packages/slicutlet.pth
	@rm -f .venv/lib/python*/site-packages/_slicutlet_editable_loader.py
	@rm -f .venv/lib/python*/site-packages/slicutlet-editable.pth
	@rm -rf .venv/lib/python*/site-packages/__pycache__/_slicutlet_editable_loader*.pyc

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(INSTALL_DIR)
	find . -path './.venv' -prune -o -type d -name '__pycache__' -print -exec rm -rf {} + 2>/dev/null || true
	find . -path './.venv' -prune -o -type f -name '*.pyc' -print -delete 2>/dev/null || true
	find python -type f -name '*.so' -delete 2>/dev/null || true
	find python -type f -name '*.dylib' -delete 2>/dev/null || true

.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf .venv

# Git workflow helpers
.PHONY: status
status:
	@echo "=== Git Status ==="
	@git status --short
	@echo ""
	@echo "=== Modified Files ==="
	@git diff --name-only

.PHONY: branch
branch:
	@git branch -vv

# Info target
.PHONY: info
info:
	@echo "=== Build Configuration ==="
	@echo "Python:       $(PYTHON)"
	@echo "Build dir:    $(BUILD_DIR)"
	@echo "Install dir:  $(INSTALL_DIR)"
	@echo ""
	@echo "=== Environment ==="
	@echo "DYLD_LIBRARY_PATH: $(DYLD_LIBRARY_PATH)"
	@echo "PYTHONPATH:        $(PYTHONPATH)"
	@echo ""
	@echo "=== Installed Packages ==="
	@$(UV) pip list | grep slicutlet || echo "slicutlet not in pip list"
