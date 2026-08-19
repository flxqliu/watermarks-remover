#!/usr/bin/env bash
# Build WatermarksRemover.app: compile the SwiftUI front end, then wrap it in a
# bundle together with the Python service it drives.
#
#   ./app/macos/Scripts/build-app.sh                 # release build
#   CONFIGURATION=debug ./app/macos/Scripts/build-app.sh
#   OUTPUT_DIR=~/Applications ./app/macos/Scripts/build-app.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
configuration="${CONFIGURATION:-release}"
output_dir="${OUTPUT_DIR:-$here/build}"
app="$output_dir/Watermarks Remover.app"
python="${PYTHON:-python3}"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "error: this app builds on macOS only (found $(uname -s))" >&2
  exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "error: swift not found. Install Xcode or the Command Line Tools." >&2
  exit 1
fi

echo "==> Building ($configuration)"
swift build --package-path "$here" -c "$configuration"
binary="$(swift build --package-path "$here" -c "$configuration" --show-bin-path)/WatermarksRemover"
test -x "$binary" || { echo "error: build produced no binary at $binary" >&2; exit 1; }

echo "==> Assembling bundle"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources/service/scripts"
cp "$binary" "$app/Contents/MacOS/WatermarksRemover"
cp "$here/Info.plist" "$app/Contents/Info.plist"
printf 'APPL????' > "$app/Contents/PkgInfo"

# The service is stdlib-only, so the bundle only needs the scripts themselves.
cp "$repo"/service/scripts/*.py "$app/Contents/Resources/service/scripts/"
cp "$repo/LICENSE" "$app/Contents/Resources/LICENSE"

echo "==> Rendering icon"
iconset="$(mktemp -d)/AppIcon.iconset"
if "$python" "$here/Scripts/make_icon.py" --out "$iconset" >/dev/null; then
  if command -v iconutil >/dev/null 2>&1; then
    iconutil -c icns -o "$app/Contents/Resources/AppIcon.icns" "$iconset"
  else
    echo "    iconutil missing; shipping without an icon"
  fi
else
  echo "    icon generation failed; shipping without an icon"
fi

# Ad-hoc signature: without it macOS re-prompts for local-network and file
# access on every rebuild, and quarantine handling is noisier.
if command -v codesign >/dev/null 2>&1; then
  echo "==> Signing (ad-hoc)"
  codesign --force --sign - --timestamp=none "$app" >/dev/null 2>&1 ||
    echo "    ad-hoc signing failed; the app still runs after right-click > Open"
fi

echo "==> Done: $app"
