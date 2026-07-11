#!/usr/bin/env bash
echo "=== Git Hygiene Scan ==="
echo

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "Not inside a git repo. Run this from your project root after 'git init'."
  exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

echo "-- .gitignore --"
if [ -f .gitignore ]; then
  echo "OK: .gitignore exists."
else
  echo "MISSING: no .gitignore found."
fi
echo

echo "-- Already-tracked sensitive/heavy paths --"
PATTERNS='venv/|\.venv/|(^|/)env/|node_modules/|(^|/)\.env($|\.[^.]*$)|__pycache__/|\.DS_Store$|\.pem$|\.key$'
TRACKED_HITS=$(git ls-files | grep -E "$PATTERNS" || true)
if [ -z "$TRACKED_HITS" ]; then
  echo "OK: none of the common sensitive/heavy patterns are tracked."
else
  echo "FOUND (already in git's index, .gitignore won't remove them):"
  echo "$TRACKED_HITS" | sed 's/^/  /'
fi
echo

echo "-- Rough secret scan (heuristic, verify manually) --"
SECRET_HITS=$(git grep -InE '(AKIA[0-9A-Z]{16}|sk-[a-zA-Z0-9]{20,}|api[_-]?key\s*=\s*["\047][^"\047]+["\047]|password\s*=\s*["\047][^"\047]+["\047]|SECRET[_A-Z]*\s*=)' -- . 2>/dev/null | grep -v '.gitignore' || true)
if [ -z "$SECRET_HITS" ]; then
  echo "OK: no obvious secret-looking strings in tracked files."
else
  echo "FLAGGED (heuristic match, could be a false positive — check each):"
  echo "$SECRET_HITS" | sed 's/^/  /'
fi
echo

echo "-- Largest tracked files --"
git ls-files -z | xargs -0 du -h 2>/dev/null | sort -rh | head -5 | sed 's/^/  /'
echo

echo "-- Path check --"
case "$REPO_ROOT" in
  *OneDrive*)
    echo "NOTE: this repo is inside a OneDrive-synced path ($REPO_ROOT)."
    ;;
  *)
    echo "OK: not inside a OneDrive-synced path."
    ;;
esac

echo
echo "=== Scan complete ==="
