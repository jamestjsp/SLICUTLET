# CLAUDE.md

SLICUTLET-specific guidance for Claude Code. See `~/.claude/CLAUDE.md` for general workflow.

## Project Overview

SLICUTLET: C11 translation of SLICOT (Systems and Control library) from Fortran77.

**Constraints:**
- Requires C11-compliant compiler (MSVC not supported - lacks `complex.h`)
- `c128` must match Fortran COMPLEX*16 layout (two f64s)
- BLAS/LAPACK with auto-detected symbol mangling

**Reference:**
- Fortran77 source: `SLICOT-Reference/src/` (git submodule)
- Update: `git submodule update --init --recursive`

## Build Commands

### Using uv (Recommended)

**Install dependencies:**
```bash
uv sync
```

**Build and run tests:**
```bash
# Build once (rebuilds when source changes)
uv run meson setup build -Dpython=true && uv run meson compile -C build && uv run meson install -C build --destdir="$(pwd)/build-install"

# If editable install exists, remove it first (avoids cp313 FileNotFoundError)
uv pip uninstall slicutlet

# Run tests (direct venv, uses build-install)
export DYLD_LIBRARY_PATH=build-install/usr/local/lib
export PYTHONPATH=build-install/usr/local/lib/python3.13/site-packages
.venv/bin/python -m pytest python/tests/ -v

# Specific tests
.venv/bin/python -m pytest python/tests/test_mb01xx.py -v
```

**Code quality:**
```bash
uv run ruff check python/                  # Check
uv run ruff check --fix python/            # Fix auto-fixable issues
uv run ruff format python/                 # Format
```

**Add dependencies:**
```bash
uv add package-name         # Runtime
uv add --dev package-name   # Dev only (ruff, pytest, etc)
```

### Manual Build with Meson

**Setup and build:**
```bash
meson setup build -Dpython=true
meson compile -C build
meson install -C build --destdir="$(pwd)/build-install"
```

**Options:**
- `-Dpython=true/false` - Build Python extension module (default: false)
- `-Dilp64=true/false` - Use ILP64 BLAS/LAPACK interface (default: false)

**C smoke test:**
```bash
meson test -C build
```

**Python tests (manual build):**
```bash
export PYTHONPATH=build-install/usr/local/lib/python3.13/site-packages
pytest python/tests/
```

## Code Structure

**Function families** (organized by SLICOT naming):
- `src/AB/` - State-space transformations (ab01nd, ab04md, ab05md, ab05nd, ab07nd)
- `src/MA/` - Matrix operations (ma01xx, ma02xx families)
- `src/MB/` - Matrix operations continued (mb01xx, mb03xx families)
- `src/MC/` - Polynomial operations (mc01xx family)
- `src/B/` - Test utilities

**Headers:**
- `src/include/types.h` - Type aliases (i32, i64, f64, c128, etc.)
- `src/include/slc_blaslapack.h` - BLAS/LAPACK declarations with detected mangling
- `src/include/slc_config.h.in` - Config template for Fortran symbol detection
- `include/SLICUTLET/slicutlet.h` - Public API

**Python bindings:**
- `python/slicutletmodule.c` - Cython extension wrapping C functions
- `python/slicutlet/__init__.py` - Python package interface
- `python/tests/` - Test suite using pytest + numpy.testing

## Translation Workflow

### Dependency Analysis

Run before planning:
```bash
uv run tools/extract_dependencies.py SLICOT-Reference/src/ AB01ND    # Specific
uv run tools/extract_dependencies.py SLICOT-Reference/src/           # Full
uv run tools/extract_dependencies.py SLICOT-Reference/src/ | grep "Level 0"  # Leaves
```

**Output:**
- Dependency levels (0=leaves, N=higher)
- LAPACK/BLAS requirements
- Reverse dependencies

**Planning:** Bottom-to-top (Level 0 → Level N)

### Parallel Translation with Worktrees

**Create worktrees:**
```bash
git worktree add ../SLICUTLET-wt1 wt1-routine-name
git worktree add ../SLICUTLET-wt2 wt2-routine-name
git worktree add ../SLICUTLET-wt3 wt3-routine-name
git worktree list
```

**Launch parallel subagents (Task tool):**
- Work in separate worktrees
- Follow RED→GREEN→REFACTOR→VERIFY cycle
- Tests FIRST, verify failure before implementing
- Report line numbers

### TDD Cycle (RED → GREEN → REFACTOR → VERIFY)

**RED:** Write test first
- Extract from `SLICOT-Reference/examples/` or docs
- Write in `python/tests/test_xxnnxx.py`
- Verify FAILS
- Commit: `RED: Add tests for xxnncc`

**GREEN:** Implement to pass
- Read Fortran: `SLICOT-Reference/src/XXNNCC.f`
- Translate to C: `src/XX/xxnncc.c` (lowercase)
- Add to `meson.build` lib_srcs
- Update `python/slicutletmodule.c` exports
- Run: `.venv/bin/python -m pytest python/tests/test_xxnnxx.py`
- Commit: `GREEN: Implement xxnncc`

**REFACTOR:** Clean up
- Run: `uv run ruff check --fix python/ && uv run ruff format python/`
- Code quality + numerical correctness
- Comment complex logic
- Commit: `REFACTOR: Clean up xxnncc`

**VERIFY:** Final validation
- Ruff: `uv run ruff check python/`
- Tests: `.venv/bin/python -m pytest python/tests/`
- Confirm no regressions

### Pre-Merge Rebase (🚨 CRITICAL)

Before merging, rebase ALL worktrees:
```bash
cd ../SLICUTLET-wt1
git fetch /Users/josephj/Workspace/SLICUTLET main:main
git rebase main
# Repeat for wt2, wt3...
cd /Users/josephj/Workspace/SLICUTLET
```

**Why critical:** Auto-drops duplicate dependencies if merged to main meanwhile. Prevents conflicts.

**Or use:** `./tools/rebase_worktrees.sh wt1-branch wt2-branch wt3-branch`

### Merge Worktrees

Sequential merge + validate after each:
```bash
git merge wt1-routine-name && uv run ruff check python/ && .venv/bin/python -m pytest python/tests/
git merge wt2-routine-name && uv run ruff check python/ && .venv/bin/python -m pytest python/tests/
git merge wt3-routine-name && uv run ruff check python/ && .venv/bin/python -m pytest python/tests/
```

Cleanup:
```bash
git worktree remove ../SLICUTLET-wt{1,2,3}
```

### Quality Checklist

- [ ] Ruff clean: `uv run ruff check python/`
- [ ] Tests pass: `.venv/bin/python -m pytest python/tests/`
- [ ] Builds: `uv run meson compile -C build`
- [ ] Min 3 tests per routine
- [ ] Test data from SLICOT reference docs
- [ ] Edge cases (N=0, singular matrices)
- [ ] TDD commits (RED→GREEN→REFACTOR)
- [ ] Worktrees rebased before merge
- [ ] Python exports updated

### Rebase Automation Script

`tools/rebase_worktrees.sh` (usage: `./tools/rebase_worktrees.sh wt1-branch wt2-branch ...`)

```bash
#!/bin/bash
MAIN_DIR="/Users/josephj/Workspace/SLICUTLET"
BASE_DIR="/Users/josephj/Workspace"

for branch in "$@"; do
    wt_num=$(echo "$branch" | grep -o 'wt[0-9]')
    wt_dir="$BASE_DIR/SLICUTLET-$wt_num"
    [ -d "$wt_dir" ] || { echo "Skip: $wt_dir"; continue; }

    cd "$wt_dir" || exit 1
    git fetch "$MAIN_DIR" main:main && git rebase main || {
        git rebase --abort
        echo "ERROR: Rebase failed for $branch"
        exit 1
    }
done

cd "$MAIN_DIR"
echo "Ready to merge: $*"
```

### Common Pitfalls

| Issue | Fix |
|-------|-----|
| Duplicate deps | Rebase worktrees before merge |
| Same insertion point | Manual merge or placeholder comments |
| Stale worktrees | Always rebase before merge |
| Test conflicts | Separate files per family |
| Missing exports | Update `slicutletmodule.c` + `__init__.py` |
| Build errors | Add to `meson.build` lib_srcs |

**File mapping:** `SLICOT-Reference/src/AB01ND.f` → `src/AB/ab01nd.c`

## Translation Conventions

**Types:** `i32`→INTEGER, `f64`→DOUBLE PRECISION, `c128`→COMPLEX*16

**Mode Parameters (CHARACTER*1 → i32):**
- Fortran `CHARACTER*1` parameters (UPLO, TRANS, SIDE, etc.) → `const i32` in C
- **Deliberate design choice** to avoid string handling in C
- Use integer values: 0, 1, 2, etc. for different modes
- Examples:
  - `UPLO`: 0=Upper, 1=Lower
  - `TRANS`: 0=NoTranspose, 1=Transpose, 2=ConjugateTranspose
  - `SIDE`: 0=Left, 1=Right
- Python tests pass integers (0, 1) not strings
- **Do NOT convert to `const char*` - this is intentional**

**Arrays:** 0-based C indexing with Fortran logic adjustments. Preserve lda/ldb/ldz for BLAS/LAPACK.

**BLAS/LAPACK:** Auto-detected symbol mangling via `slc_blaslapack.h` macros.

## Test Strategy

**Validation:**
- Numerical correctness vs known results
- Edge cases (N=0, singular matrices)
- `numpy.testing.assert_allclose` (rtol=1e-14)

**Structure:**
- File per family: `test_ma01xx.py`, `test_mb01xx.py`
- Classes: `TestMA01AD`, `TestMA01BD`
- Descriptive method names
