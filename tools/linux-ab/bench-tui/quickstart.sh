#!/bin/bash
# One-command build + run for bench-tui, on any Linux/WSL with g++ and cmake.
# Usage: tools/linux-ab/bench-tui/quickstart.sh [extra bench-tui args]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$HERE/build}
cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc 2>/dev/null || echo 4)"
exec "$BUILD/bench-tui" "$@"
