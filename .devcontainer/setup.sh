#!/bin/bash

echo "Setting up SLICUTLET dev environment..."

# Install Python dependencies with uv
echo "Installing Python dependencies with uv..."
uv sync || {
    echo "WARNING: uv sync failed - you may need to run it manually"
    exit 0
}

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Setting up initial build..."
    uv run meson setup build -Dpython=true || {
        echo "WARNING: Meson setup failed - run manually: uv run meson setup build -Dpython=true"
        exit 0
    }
    uv run meson compile -C build || {
        echo "WARNING: Meson compile failed - run manually: uv run meson compile -C build"
        exit 0
    }
    uv run meson install -C build --destdir="$(pwd)/build-install" || {
        echo "WARNING: Meson install failed"
        exit 0
    }
    echo "Initial build complete"
else
    echo "Build directory exists - skipping initial build"
    echo "Run: uv run meson compile -C build && uv run meson install -C build --destdir=\$(pwd)/build-install"
fi

# Verify installation
echo "Verifying tools..."
gcc --version | head -n1
gfortran --version | head -n1
meson --version
uv --version
python3 --version

echo ""
echo "Dev container ready!"
echo ""
echo "Quick commands:"
echo "  Build:  uv run meson compile -C build && uv run meson install -C build --destdir=\$(pwd)/build-install"
echo "  Test:   uv run pytest python/tests/"
echo "  Lint:   uv run ruff check python/"
echo "  Format: uv run ruff format python/"
echo ""
