#!/bin/bash
# Sketch Guide - PC 단독 실행
# Chrome 앱 모드로 standalone.html을 브라우저 UI 없이 실행

CHROME="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HTML_PATH="$(wslpath -w "$SCRIPT_DIR/standalone.html")"

if [ ! -f "$CHROME" ]; then
  echo "Error: Chrome not found at $CHROME"
  exit 1
fi

if [ ! -f "$SCRIPT_DIR/standalone.html" ]; then
  echo "Error: standalone.html not found. Run 'node scripts/build-standalone.js' first."
  exit 1
fi

echo "Starting Sketch Guide..."
"$CHROME" --app="file:///$HTML_PATH" --start-maximized &
