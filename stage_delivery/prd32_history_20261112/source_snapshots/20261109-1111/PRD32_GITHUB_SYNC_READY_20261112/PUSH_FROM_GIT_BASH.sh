#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo 'Usage: bash PUSH_FROM_GIT_BASH.sh /path/to/local/fashion_multimodal_analysis' >&2
  exit 2
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$1" && pwd)"
BASE='stage/prd32-20261027-closeout'
BRANCH='work/prd32-20261112-audit'

if [[ "$(git -C "$REPO" rev-parse --is-inside-work-tree)" != true ]]; then
  echo 'Not a Git work tree' >&2; exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo 'Working tree is not clean. Save or stash your local changes first.' >&2; exit 2
fi
remote="$(git -C "$REPO" remote get-url origin)"
case "$remote" in
 *Selina-aurora/fashion_multimodal_analysis* ) ;;
 * ) echo "Refusing unexpected origin remote: $remote" >&2; exit 2 ;;
esac

git -C "$REPO" fetch origin
if ! git -C "$REPO" show-ref --verify --quiet 'refs/remotes/origin/stage/prd32-20261027-closeout'; then
 echo 'Expected base branch missing on remote. No changes made.' >&2; exit 3
fi
if git -C "$REPO" show-ref --verify --quiet "refs/heads/$BRANCH"; then
 echo "Branch already exists locally: $BRANCH. No changes made." >&2; exit 3
fi
if git -C "$REPO" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
 echo "Branch already exists on remote: $BRANCH. No changes made." >&2; exit 3
fi

git -C "$REPO" switch -c "$BRANCH" "origin/$BASE"
cp -a "$HERE/UPLOAD_TO_REPO/." "$REPO/"
git -C "$REPO" add -- docs/BRANCH_STATUS_20261112.md stage_delivery/prd32_20261112

git -C "$REPO" diff --cached --check
if git -C "$REPO" diff --cached --quiet; then
 echo 'No staged differences; nothing to push.'; exit 0
fi

echo 'Files staged:'
git -C "$REPO" diff --cached --stat

git -C "$REPO" commit -m 'docs(prd32): add Nov 11 experiment evidence and reproduction snapshot'
git -C "$REPO" push -u origin "$BRANCH"
echo 'PUSH_OK: work/prd32-20261112-audit (not merged into main)'
echo 'Review: https://github.com/Selina-aurora/fashion_multimodal_analysis/compare/main...work/prd32-20261112-audit?expand=1'
