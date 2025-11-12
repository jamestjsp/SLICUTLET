# SLICUTLET Dev Container

Development container for SLICUTLET with all required build tools and dependencies.

## What's Included

### Build Tools
- **gcc** - C11 compiler
- **gfortran** - Fortran compiler for SLICOT reference
- **make, cmake** - Build systems
- **meson, ninja** - Primary build system for SLICUTLET

### Libraries
- **BLAS/LAPACK** - Linear algebra libraries (OpenBLAS)
- **libblas-dev, liblapack-dev** - Development headers

### Python Environment
- **Python 3.13** - Latest Python
- **uv** - Fast Python package manager
- **pytest** - Testing framework
- **ruff** - Linting and formatting
- **numpy** - Numerical computing

### Development Tools
- **git** - Version control with submodule support
- **gh** - GitHub CLI
- **VSCode extensions** - Python, C++, Meson, Copilot

## Usage

### Opening in VSCode

1. Install "Dev Containers" extension
2. Open command palette (Cmd/Ctrl+Shift+P)
3. Select "Dev Containers: Reopen in Container"
4. Wait for container build and setup (may take 2-3 minutes on first run)

### Initial Setup

Container automatically runs `.devcontainer/setup.sh`:
- Initializes git submodules
- Installs Python dependencies with uv
- Builds project with meson
- Verifies all tools

### Development Workflow

**Build:**
```bash
uv run meson compile -C build
uv run meson install -C build --destdir=$(pwd)/build-install
```

**Test:**
```bash
PYTHONPATH=build-install/usr/local/lib/python3.13/site-packages \
DYLD_LIBRARY_PATH=build-install/usr/local/lib \
uv run pytest python/tests/
```

**Lint and format:**
```bash
uv run ruff check python/
uv run ruff format python/
```

**Add dependencies:**
```bash
uv add package-name          # Runtime
uv add --dev package-name    # Dev only
```

### Environment Variables

Pre-configured in devcontainer.json:
- `PYTHONPATH` - Python module search path
- `LD_LIBRARY_PATH` - Shared library path (Linux)
- `DYLD_LIBRARY_PATH` - Shared library path (macOS host)

### Rebuilding Container

If you modify Dockerfile or devcontainer.json:
1. Command palette → "Dev Containers: Rebuild Container"
2. Or "Dev Containers: Rebuild Without Cache"

## Troubleshooting

**Container fails to start:**
1. Command palette → "Dev Containers: Rebuild Container Without Cache"
2. Check Docker Desktop is running
3. Remove old containers: `docker ps -a | grep slicutlet` then `docker rm <container_id>`

**Setup script failed:**
Setup script is non-blocking - container will still start. Run manually:
```bash
bash .devcontainer/setup.sh
```

**Submodules not initialized:**
```bash
git submodule update --init --recursive
```

**Build directory issues:**
```bash
rm -rf build build-install
uv run meson setup build -Dpython=true
uv run meson compile -C build
uv run meson install -C build --destdir=$(pwd)/build-install
```

**Python path issues:**
Already set in devcontainer.json. If needed manually:
```bash
export PYTHONPATH=build-install/usr/local/lib/python3.13/site-packages
export LD_LIBRARY_PATH=build-install/usr/local/lib
```

**uv command not found:**
Rebuild container - uv should be in `/usr/local/bin`

## Architecture Notes

- **C11 required** - MSVC not supported (lacks `complex.h`)
- **Symbol mangling** - BLAS/LAPACK auto-detected at build time
- **ILP64** - Optional 64-bit integer BLAS interface (`-Dilp64=true`)
- **Fortran layout** - `c128` must match COMPLEX*16 (two f64s)
