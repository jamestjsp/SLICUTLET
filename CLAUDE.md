# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLICUTLET: C11 translation of SLICOT (Systems and Control library) from Fortran77. Provides easier integration with modern languages by eliminating Fortran ABI binding difficulties.

**Constraints:**
- Requires C11-compliant compiler (MSVC not supported - lacks proper `complex.h` support)
- Complex type `c128` must match Fortran COMPLEX*16 layout (two f64s)
- Depends on BLAS/LAPACK with auto-detected symbol mangling (lowercase+underscore, lowercase, or uppercase)

**Reference Code:**
- Original Fortran77 source: `SLICOT-Reference/src/` (git submodule)
- Clone with submodule: `git clone --recurse-submodules <repo>`
- Update submodule: `git submodule update --init --recursive`

## Build Commands

### Using uv (Recommended)

**Install dependencies:**
```bash
uv sync
```

**Run tests:**
```bash
uv run pytest python/tests/                                                          # All tests
uv run pytest -v python/tests/test_mb01xx.py                                         # Single file
uv run pytest -v python/tests/test_ma01xx.py::TestMA01AD::test_positive_real_positive_imag  # Specific test
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

**Finding source files:**
```bash
# Original Fortran in SLICOT-Reference/src/
ls SLICOT-Reference/src/AB*.f    # AB family
ls SLICOT-Reference/src/MA*.f    # MA family
```

**Translation steps:**
1. Read Fortran source from `SLICOT-Reference/src/XXNNCC.f`
2. Translate to C in `src/XX/xxnncc.c` (lowercase naming)
3. Add to `meson.build` lib_srcs list
4. Write Python test in `python/tests/test_xxnnxx.py`
5. Update Python module exports if needed

**Example mapping:**
- `SLICOT-Reference/src/AB01ND.f` → `src/AB/ab01nd.c`
- `SLICOT-Reference/src/MA02BD.f` → `src/MA/ma02bd.c`

## Translation Conventions

**C to Fortran type mapping:**
- `i32` → INTEGER (or `i64` with -Dilp64)
- `f64` → DOUBLE PRECISION
- `c128` → COMPLEX*16 (verified at compile-time)

**Array indexing:**
- C uses 0-based indexing; translations maintain Fortran logic with index adjustments
- Leading dimension parameters (lda, ldb, ldz) preserved for BLAS/LAPACK compatibility

**BLAS/LAPACK calls:**
- Build system auto-detects symbol mangling via compile-time probes
- Macros in `slc_blaslapack.h` adapt to detected convention

## Test Strategy

**Python tests validate:**
- Numerical correctness against known results
- Edge cases (zero matrices, singular cases)
- Error handling (via pytest.raises)
- Uses `numpy.testing.assert_allclose` with tight tolerances (rtol=1e-14)

**Test structure:**
- One file per function family (test_ma01xx.py, test_mb01xx.py, etc.)
- Classes group related tests (TestMA01AD, TestMA01BD, etc.)
- Descriptive method names indicating test scenario
