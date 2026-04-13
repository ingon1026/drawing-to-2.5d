#!/bin/bash
# Sketch Guide - Launch as standalone app window
# Tries Chrome/Chromium in app mode (no browser UI)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HTML="$SCRIPT_DIR/standalone.html"

if [ ! -f "$HTML" ]; then
  echo "Error: standalone.html not found."
  echo "Run 'node scripts/build-standalone.js' first to build it."
  exit 1
fi

# Detect platform and Chrome location
if [ -f "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" ]; then
  # WSL2 → Windows Chrome
  WIN_PATH="$(wslpath -w "$HTML")"
  "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --app="file:///$WIN_PATH" --start-maximized &
elif command -v google-chrome &>/dev/null; then
  google-chrome --app="file://$HTML" --start-maximized &
elif command -v chromium-browser &>/dev/null; then
  chromium-browser --app="file://$HTML" --start-maximized &
elif command -v chromium &>/dev/null; then
  chromium --app="file://$HTML" --start-maximized &
else
  echo "Chrome/Chromium not found. Open manually:"
  echo "  $HTML"
  exit 1
fi

echo "Sketch Guide launched."
