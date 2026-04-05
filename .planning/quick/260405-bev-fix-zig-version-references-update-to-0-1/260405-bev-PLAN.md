---
phase: quick
plan: 260405-bev
type: execute
wave: 1
depends_on: []
files_modified:
  - CLAUDE.md
  - .github/workflows/test.yml
  - src/inference/tokenizer.zig
  - README.md
autonomous: true
requirements: []

must_haves:
  truths:
    - "CLAUDE.md no longer misleads AI agents with 'Do NOT use Zig master (0.14+)'"
    - "CLAUDE.md states Zig 0.15.2 as the pinned truth throughout"
    - "httpz references in CLAUDE.md say 'master branch' not 'zig-0.13 branch'"
    - "CI workflow uses ZIG_VERSION: 0.15.2"
    - "README.md build dependencies list Zig 0.15.2"
    - "tokenizer.zig comment no longer says 'needs porting from Zig 0.13 to 0.15'"
  artifacts:
    - path: "CLAUDE.md"
      provides: "Accurate tech stack guidance with Zig 0.15.2 as pinned truth"
    - path: ".github/workflows/test.yml"
      provides: "CI that uses ZIG_VERSION: 0.15.2"
    - path: "README.md"
      provides: "Correct build dependency listing"
    - path: "src/inference/tokenizer.zig"
      provides: "Accurate comment about porting status"
  key_links:
    - from: "CLAUDE.md"
      to: "AI agents reading it"
      via: "Tech stack table"
      pattern: "Zig.*0\\.15\\.2"
---

<objective>
Update all Zig version references from 0.13.0 to 0.15.2 and httpz from zig-0.13 branch to master branch across CLAUDE.md, the CI workflow, README.md, and tokenizer.zig.

Purpose: The project already runs on Zig 0.15.2 (build.zig.zon is correct), but stale 0.13.0 references confuse AI agents into making wrong decisions — such as adding zig-0.13 branch httpz URLs that don't work.

Output: Four files updated. CLAUDE.md is the primary target; it becomes the single source of truth that agents read first.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@build.zig.zon
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update CLAUDE.md — fix all Zig and httpz version references</name>
  <files>CLAUDE.md</files>
  <action>
Rewrite the stale version entries inside the GSD:stack-start block. Do NOT change any other section (Constraints, Architecture, Conventions, GSD Workflow, Developer Profile). Make only targeted edits:

1. Tech stack table — Zig row (line 26):
   - Change version from `0.13.0` to `0.15.2`
   - Change why-recommended text: remove "Do NOT use Zig master (0.14+) — MLX.zig and httpz zig-0.13 branch are not compatible with it yet." Replace with: "Installed at /opt/homebrew/bin/zig. Zig 0.15.2 is the pinned truth for this project."

2. Tech stack table — httpz row (line 29):
   - Change version from `zig-0.13 branch (master targets Zig 0.15.1)` to `master branch`
   - Change why-recommended: remove "Must use the `zig-0.13` branch archive URL, not master." Replace with: "Use the master branch archive URL (already correct in build.zig.zon)."

3. httpz Handler Pattern heading (line 47):
   - Change `httpz — Handler Pattern (zig-0.13 branch)` to `httpz — Handler Pattern (master branch)`

4. std.json and std.http.Client rows (lines 34-35):
   - Change `stdlib (Zig 0.13.0)` to `stdlib (Zig 0.15.2)` in both rows

5. Installation / Build Setup comment (line 60):
   - Change `# Zig 0.13.0 required -- verify:` to `# Zig 0.15.2 required -- verify:`

6. Alternatives Considered table (lines 70-74):
   - Change all three `httpz zig-0.13 branch` entries in the Recommended column to `httpz master branch`
   - Change the when-to-use cell for the first row: remove "Only when project upgrades to Zig 0.14+" and replace with "N/A — master branch is already in use"

7. What NOT to Use table (lines 78-83):
   - Remove the row: `| httpz \`master\` branch | Targets Zig 0.15.1; incompatible with Zig 0.13.0 | httpz \`zig-0.13\` branch |`
   - Remove the row: `| Zig 0.14+ or Zig master | MLX.zig targets 0.13.0; breaking changes in build API | Stay on Zig 0.13.0 until MLX.zig updates |`
   - Keep all other rows in the table unchanged

8. Version Compatibility table (lines 87-90):
   - Change `Zig 0.13.0 | MLX.zig 0.0.0, httpz zig-0.13 branch, pcre2-10.45` to `Zig 0.15.2 | MLX.zig 0.0.0, httpz master branch, pcre2-10.45`
   - Change notes: remove "Lock this combination; do not upgrade any piece without testing the others" → keep but update to "This combination is verified working. build.zig.zon pins httpz master."
   - Change `httpz master | Zig 0.15.1 only | Do not use with Zig 0.13.0` to `httpz master | Zig 0.15.2 | In use — correct branch`

9. Sources section (line 98):
   - Change `httpz zig-0.13 routing/handler pattern` to `httpz master routing/handler pattern`
  </action>
  <verify>
    <automated>grep -n "0\.13" /Users/gleicon/code/zig/zlx/CLAUDE.md | grep -v "mlx-c v0.1.2\|mlx-c-0.1.2\|v0.1.2\|#.*0\.13\|zig-0\.13.*branch.*reference\|mlx.*0\.1" | head -20</automated>
  </verify>
  <done>
No remaining "0.13" references in CLAUDE.md except mlx-c version pins (v0.1.2) which are correct and intentional. httpz references all say "master branch". Zig version reads "0.15.2" everywhere.
  </done>
</task>

<task type="auto">
  <name>Task 2: Update CI workflow, README.md, and tokenizer.zig comment</name>
  <files>.github/workflows/test.yml, README.md, src/inference/tokenizer.zig</files>
  <action>
Three targeted single-line fixes:

1. `.github/workflows/test.yml` line 28:
   - Change `ZIG_VERSION: 0.13.0` to `ZIG_VERSION: 0.15.2`

2. `README.md` line 171 (Dependencies section):
   - Change `Zig 0.13.0 (not 0.14+ — MLX.zig targets 0.13.0 specifically)` to `Zig 0.15.2`

3. `src/inference/tokenizer.zig` lines 1-4 (module docstring):
   - Change the comment block from:
     ```
     //! tokenizer.zig - Minimal tokenizer stub for Zig 0.15 compatibility
     //!
     //! This is a simplified tokenizer that provides the basic interface.
     //! The full MLX.zig tokenizer needs porting from Zig 0.13 to 0.15.
     ```
   - To:
     ```
     //! tokenizer.zig - Minimal tokenizer stub
     //!
     //! This is a simplified tokenizer that provides the basic interface.
     //! Ported to Zig 0.15.2. Uses stdlib JSON and file I/O directly.
     ```
  </action>
  <verify>
    <automated>grep -n "0\.13" /Users/gleicon/code/zig/zlx/.github/workflows/test.yml /Users/gleicon/code/zig/zlx/README.md /Users/gleicon/code/zig/zlx/src/inference/tokenizer.zig</automated>
  </verify>
  <done>
grep returns no matches — all three files are free of 0.13 references.
  </done>
</task>

</tasks>

<verification>
Run both verify commands after execution:

```bash
# Task 1 — no stale 0.13 in CLAUDE.md (mlx-c pins excepted)
grep -n "0\.13" /Users/gleicon/code/zig/zlx/CLAUDE.md | grep -v "mlx-c v0.1.2\|mlx-c-0.1.2\|v0.1.2"

# Task 2 — no 0.13 in CI, README, tokenizer
grep -n "0\.13" \
  /Users/gleicon/code/zig/zlx/.github/workflows/test.yml \
  /Users/gleicon/code/zig/zlx/README.md \
  /Users/gleicon/code/zig/zlx/src/inference/tokenizer.zig

# Confirm 0.15.2 pinned in key locations
grep -n "0\.15\.2" \
  /Users/gleicon/code/zig/zlx/CLAUDE.md \
  /Users/gleicon/code/zig/zlx/.github/workflows/test.yml \
  /Users/gleicon/code/zig/zlx/README.md
```

Expected: no 0.13 hits (beyond mlx-c pins), at least 3 lines confirming 0.15.2.
</verification>

<success_criteria>
- CLAUDE.md: Zig version reads 0.15.2; httpz says "master branch"; no "Do NOT use Zig master" warning; "What NOT to Use" table has no httpz-master or Zig-0.14+ rows
- .github/workflows/test.yml: ZIG_VERSION is 0.15.2
- README.md: Build dependencies list Zig 0.15.2 with no parenthetical about 0.14+
- src/inference/tokenizer.zig: Comment says "Zig 0.15.2" with no "needs porting" language
- build.zig.zon: Unchanged (already correct — minimum_zig_version = "0.15.2", httpz master URL)
</success_criteria>

<output>
After completion, create `.planning/quick/260405-bev-fix-zig-version-references-update-to-0-1/260405-bev-SUMMARY.md` with what was changed in each file.
</output>
