# NewTypeEngine tools

Python utilities shipped alongside the engine. Pure stdlib; no install step.

| File | Purpose |
|------|---------|
| `generate_project.py` | Scaffold a new NewTypeEngine VS2022 project from `src/Template.cpp`. |
| `cinder_blocks.py` | Add / remove / list / update Cinder blocks **and engine addons** in an existing project. |
| `shader_generator/` | Runtime shader codegen helpers. |
| `test_cinder_blocks.py` | Tests for `cinder_blocks.py`. Run with `python -m unittest tools.test_cinder_blocks`. |

---

## Engine addons (`engine_addons/`)

Optional engine features packaged as cinderblock.xml manifests, managed by
`cinder_blocks.py` alongside real Cinder blocks. Each lives under
`engine_addons/<name>/cinderblock.xml`.

| Addon | Folder | What it does |
|-------|--------|--------------|
| `sim` | `engine_addons/sim` | Links **LuisaComputeSimulator** (physics solver). Adds `LCSRoot`/`LCSInclude*`/`LCSLib*` MSBuild properties to `.props`, appends the LCS include paths + `LCS_NO_INTERNAL_FIBER_SCHEDULER` preprocessor define to all 3 configs, links `luisa-compute-solver-lib.lib`, and registers `Physics.cpp/.h` from `..\src\newtype\physics\` and `..\include\newtype\physics\`. |
| `video` | `engine_addons/video` | Wires the **D3D11↔D3D12 video decode pipeline** (Media Foundation → Luisa image). Adds `d3d11.lib;dxguid.lib;mf.lib;mfplat.lib;mfreadwrite.lib;mfuuid.lib` link inputs and registers the three sources + four headers under `..\src\newtype\media\` and `..\include\newtype\media\`. (The runtime gate `NT_ENABLE_MEDIA_PLAYER` lives in `Config.h` — flip it to 1 to actually enable decode.) |
| `directml` | `engine_addons/directml` | Wires **WinML / DirectML** dependencies for ONNX-based inference workloads (hand tracking, segmentation, etc.). Links `runtimeobject.lib`, adds `/bigobj` to `<AdditionalOptions>`, and sets `<CppWinRTEnabled>true</CppWinRTEnabled>` in Globals so MSBuild runs `cppwinrt.exe`. Does NOT include any inference classes — bring your own. |

```bash
# Add to a freshly-generated project:
python tools/cinder_blocks.py add sim       --project D:/Projects/MyDemo
python tools/cinder_blocks.py add video     --project D:/Projects/MyDemo
python tools/cinder_blocks.py add directml  --project D:/Projects/MyDemo

# List everything installable (Cinder blocks + engine addons):
python tools/cinder_blocks.py list --available
```

### Schema extensions to cinderblock.xml

Engine addons use three small additive schema extensions on top of the
standard Cinder block format. None of them affect existing Cinder block
manifests (all 83 original tests still pass).

| Extension | Purpose |
|-----------|---------|
| `absolute="true"` on `<source>` / `<header>` / `<includePath>` | Emit the path verbatim instead of anchoring at `$(CinderBlocksDir)\<folder>\...`. Used for in-engine paths (`..\src\newtype\physics\Physics.cpp`) and engine MSBuild macros (`$(LCSInclude)`). Mirrors the existing `<staticLibrary absolute="true">` semantics. |
| `<property name="X">VALUE</property>` (inside `<block>`) | Emit a new MSBuild `<X>VALUE</X>` line into the project's `.props`. The lines live in a marker region (`REGION_PROPS_PROPERTIES`) so they round-trip cleanly with `add` / `remove` / `unscaffold`. |
| `<preprocessorDefine>` now actually emitted | Previously parsed but never wired. Now flows into a new scaffold macro `$(CinderBlocksDefines)`, injected into `<PreprocessorDefinitions>` in all 3 configs (Debug, Debug_Runtime, Release). |
| `<additionalOption>VALUE</additionalOption>` | Append VALUE (e.g. `/bigobj`) to `<AdditionalOptions>` in every config. Space-joined via the `$(CinderBlocksAdditionalOptions)` scaffold macro. |
| `<globalsProperty name="X">VALUE</globalsProperty>` | Emit `<X>VALUE</X>` inside the vcxproj's `<PropertyGroup Label="Globals">` (project-level flags like `<CppWinRTEnabled>true</CppWinRTEnabled>`). Lives in a marker region `REGION_VCXPROJ_GLOBALS`. |

### Discovery

`BlockRegistry` walks both the Cinder blocks directory (default
`C:\Users\barca\Projects\Cinder\blocks`, override via `--blocks-dir` or
`CINDER_BLOCKS_DIR`) **and** the engine addons directory (default
`<engine>/engine_addons/`, override via `--engine-addons-dir` or
`NT_ENGINE_ADDONS_DIR`). The Cinder dir is required; the engine addons dir
is purely additive and silently skipped if missing (so a fresh clone without
the dir still works for Cinder-only flows).

The default MUST match the `<CinderRoot>` declared in `vc2022/*.props`:
generated vcxproj paths anchor at `$(CinderBlocksDir) = $(CinderRoot)\blocks`,
so a scan/reference mismatch would resolve to a different on-disk tree at
build time.

First-seen wins on folder-name collisions, with the Cinder blocks dir taking
precedence over `engine_addons/`.

### Migration note (engine dev project)

`generate_project.py` reads from `vc2022/EngineTemplate.props` — a minimal
template containing only core engine dependencies (Cinder + LuisaCompute).
Optional features like LCS / video are not baked in; install them on demand
via `cinder_blocks.py add`.

The engine's own dev project at `vc2022/NewTypeEngine.props` /
`NewTypeEngine.vcxproj` is separate and currently has LCS wired in directly
(Suits branch state) so the engine itself builds without needing the addon.
That's intentional — the dev project is just another NewTypeEngine consumer
and may use either approach. A future cleanup could strip LCS from the dev
project too and have it use `add sim` like any other consumer; track as a
follow-up.

---

## `cinder_blocks.py` — Cinder block manager

A from-scratch port of Cinder's TinderBox block-management logic, adapted for NewTypeEngine's existing `.vcxproj` layout (uses `$(CinderRoot)` macros defined in `<project>.props`, supports Debug / Debug_Runtime / Release configs).

### What it does

- **Parses `cinderblock.xml`** from any Cinder blocks tree — every observed XML variant is supported: `<source>` vs `<sourcePattern>` (with glob expansion), `<header>` vs `<headerPattern>`, `<includePath cinder="true">`, `<staticLibrary absolute="true">`, nested `<platform os="msw" config="debug|release">`, `<framework>` (filtered out on MSW), `<buildCopy>`, `<copyExclude>`, `<preprocessorDefine>`, `<requires>` (transitive deps).
- **Resolves dependencies transitively** with cycle detection — adding `TUIO` automatically pulls in `OSC`.
- **Mutates the project in place** via marker-delimited regions in `.vcxproj` / `.vcxproj.filters` / `.props`. Round-trip byte-identity is guaranteed: `scaffold → unscaffold` produces bytes identical to a fresh `generate_project.py` output.
- **Tracks state in a sidecar JSON** (`<project>/vc2022/.cinder-blocks.json`) — the single source of truth for which blocks are installed and in what mode.
- **Two install modes** per block:
  - **reference** (default): paths anchor at `$(CinderBlocksDir)` (=``$(CinderRoot)\blocks``); no files are copied.
  - **copy**: block files duplicated into `<project>/blocks/<folder>/`; paths anchor at `$(CinderBlocksLocalDir)` (=`..\blocks`). Honors `<copyExclude>` and always excludes `.git`/`.svn`/`.hg`.

### Prerequisites

- Python 3.10+ (uses `@dataclass` slots, PEP 604 unions).
- The default Cinder blocks tree is `C:\Users\barca\Projects\Cinder\blocks`. Override per-invocation with `--blocks-dir` or set `CINDER_BLOCKS_DIR` in the environment. This matches `$(CinderRoot)` in `vc2022/*.props`.

### Commands

#### `list` — show installed and/or available blocks

```bash
# Blocks already installed in a project (reads .cinder-blocks.json):
python tools/cinder_blocks.py list --project D:/Projects/MyDemo

# All blocks in the blocks tree, marking installed ones with [+]:
python tools/cinder_blocks.py list --available

# Both at once (the default when no --project):
python tools/cinder_blocks.py list
```

`list --available` filters to MSW-supported blocks; macOS/iOS-only blocks are listed under "Excluded". Parse errors (e.g. duplicate block IDs) are reported separately.

#### `add` — install one or more blocks

```bash
# Reference mode (default): paths use $(CinderBlocksDir)\...
python tools/cinder_blocks.py add Cinder-OSC --project D:/Projects/MyDemo

# Multiple at once — deps resolve transitively:
python tools/cinder_blocks.py add TUIO --project D:/Projects/MyDemo
#   + OSC    (auto: dependency:TUIO)  [reference]
#   + TUIO                                [reference]

# Copy mode: duplicate block files into <project>/blocks/
python tools/cinder_blocks.py add Watchdog --project D:/Projects/MyDemo --mode copy

# Dry run (no writes):
python tools/cinder_blocks.py add Cinder-NDI --project D:/Projects/MyDemo --dry-run
```

Flags:
- `--project PATH` (required) — path to the project root (must contain `vc2022/`).
- `--mode reference|copy` (default `reference`).
- `--dry-run` — print the planned changes without writing.

Auto-installed dependencies are tagged with `explicit: false` and `required_by: [<requiring block>]` in the sidecar JSON.

#### `remove` — uninstall blocks

```bash
# Remove one block; orphaned auto-deps are removed too:
python tools/cinder_blocks.py remove TUIO --project D:/Projects/MyDemo
#   - TUIO
#   - OSC  (was auto, now orphaned)

# Keep deps even if they become orphans:
python tools/cinder_blocks.py remove TUIO --project D:/Projects/MyDemo --keep-deps

# Force-remove a block that other installed blocks still depend on:
python tools/cinder_blocks.py remove OSC --project D:/Projects/MyDemo --force

# Copy-mode blocks have their <project>/blocks/<folder>/ deleted automatically.
```

Refuses by default if any installed block still depends on the target. Use `--force` to override, `--keep-deps` to preserve orphaned deps.

#### `update` — re-derive project files from the sidecar

```bash
python tools/cinder_blocks.py update --project D:/Projects/MyDemo
```

Re-reads every installed block's `cinderblock.xml` and rebuilds the `.vcxproj` / `.filters` region contents. Use cases:
- You pulled latest Cinder-LDLidar and it has new source files → `update` picks them up.
- You moved the blocks tree → update `cinder_blocks_dir` in the sidecar JSON, then `update`.
- For copy-mode blocks: re-syncs the local copy (deletes and re-copies) so removed/renamed files upstream propagate.

No-op when nothing has changed ("Already in sync.").

#### `resolve` — debug helper

```bash
python tools/cinder_blocks.py resolve TUIO
#   OSC    (dependency:TUIO)
#   TUIO   (requested)
```

Prints the topological dependency closure of the named blocks. Useful for checking what `add` would pull in without actually installing.

### Path strategy

| Element | Reference mode | Copy mode |
|---------|---------------|-----------|
| `<sourcePattern>` / `<source>` files | `$(CinderBlocksDir)\<folder>\...` | `$(CinderBlocksLocalDir)\<folder>\...` |
| `<headerPattern>` / `<header>` files | same | same |
| `<includePath>` (block-local) | appended to `$(CinderBlocksIncludePaths)` | same, using local anchor |
| `<includePath cinder="true">` | `$(CinderRoot)\<path>` | `$(CinderRoot)\<path>` (Cinder tree is always external) |
| `<staticLibrary>` (relative) | `$(CinderBlocks...Libs)` macros, full path | same, using local anchor |
| `<staticLibrary absolute="true">` | emitted verbatim (e.g. `libpng.lib`, `-lz`) | same |
| `<libraryPath>` | appended to `$(CinderBlocksLibPaths)` | same, using local anchor |
| `<buildCopy>` (DLLs) | added as `xcopy` line to post-build event | same, using local anchor |

Config-specific routing (via `<platform config="debug|release">`):
- Debug-only libs → `$(CinderBlocksDebugLibs)` (used by Debug + Debug_Runtime configs)
- Release-only libs → `$(CinderBlocksReleaseLibs)`
- Universal libs (no config) → both pipelines

### How the vcxproj gets modified

A one-time `scaffold` (run automatically on the first `add`) inserts:

1. A `PropertyGroup` with the 6 cinder-blocks MSBuild properties (`CinderBlocksIncludePaths`, `CinderBlocksLibPaths`, `CinderBlocksDebugLibs`, `CinderBlocksReleaseLibs`, `CinderBlocksPostBuild`, `CinderBlocksPostBuildChained`). Tagged with `<!-- BEGIN cinder-blocks properties -->` markers.
2. Empty `<ItemGroup>` regions for `ClCompile` and `ClInclude` (filled in by `add`).
3. References to those macros appended to the existing `<AdditionalIncludeDirectories>`, `<AdditionalDependencies>`, `<PostBuildEvent><Command>` elements. The references are sentinel-tagged (the unique macro names) so removal is exact.
4. A `<CinderBlocksDir>` and `<CinderBlocksLocalDir>` line in `<project>.props`.

After that, every `add` / `remove` / `update` only rewrites the **property values** and the **contents of the marked ItemGroups** — the rest of the vcxproj is never touched. The `remove` command leaves the scaffolding in place (cheaper for next `add`); call `unscaffold` (via the Python API or `ProjectEditor(...).unscaffold()`) to strip it for a fully clean project.

### Sidecar JSON schema

```json
{
  "schema_version": 1,
  "cinder_blocks_dir": "C:/Users/barca/Projects/Cinder/blocks",
  "blocks": {
    "OSC": {
      "block_id": "org.libcinder.osc",
      "mode": "reference",
      "explicit": true,
      "required_by": []
    },
    "TUIO": {
      "block_id": "org.libcinder.tuio",
      "mode": "copy",
      "explicit": true,
      "required_by": []
    }
  }
}
```

- `mode` — `"reference"` or `"copy"`.
- `explicit` — `true` if added directly via CLI, `false` if auto-pulled as a dependency.
- `required_by` — list of block folder names that pulled this in. Drives auto-removal when an explicit removal would orphan a dep.

The file is committed to the project repo. `cinder_blocks_dir` is the only machine-local bit; on a fresh clone, opening the project in VS still works (the macros resolve at build time, so a missing blocks dir just produces build errors rather than corrupting the project).

### Programmatic API

The same building blocks used by the CLI are importable for custom scripts:

```python
from tools import cinder_blocks as cb

# Parse one block:
m = cb.parse_manifest(Path(r"C:\Users\barca\Projects\Cinder\blocks\OSC"))

# Scan a whole blocks tree:
reg = cb.BlockRegistry(Path(r"C:\Users\barca\Projects\Cinder\blocks")).scan()
osc = reg.find("org.libcinder.osc")  # by id, name, or folder (case-insensitive)

# Resolve a dependency closure:
closure = cb.resolve_deps(["TUIO"], reg)

# Mutate a project:
editor = cb.ProjectEditor(Path("vc2022/MyDemo.vcxproj")).load()
editor.scaffold()        # one-time injection
editor.unscaffold()      # full reversal

state = cb.ProjectState(Path("."), reg).load()
state.add_block("OSC", mode=cb.MODE_REFERENCE, explicit=True)
state.apply()            # write vcxproj/filters/props regions
state.save_sidecar()
```

### Tests

```bash
python -m unittest tools.test_cinder_blocks -v
```

83 tests across 5 phases: parser edge cases (every real cinderblock.xml variant), registry lookup, dependency resolution, scaffold/unscaffold byte-identity, full CLI flow (add / remove / update / copy mode).

---

## `generate_project.py` — project scaffolder

Creates a new VS2022 project from `src/Template.cpp`.

```bash
python tools/generate_project.py --path D:/Projects --name MyDemo
#   Created: D:/Projects/MyDemo/vc2022/MyDemo.vcxproj
#   ...
```

Generates:
- `vc2022/MyDemo.sln`, `MyDemo.vcxproj`, `MyDemo.vcxproj.filters`, `MyDemo.props`
- `src/MyDemoApp.cpp` (renamed from `Template.cpp`)
- `include/Resources.h`, `resources/icon.ico`, `assets/{models,textures}/`
- `Resources.rc`

The new project references the engine source via relative paths (`../../src/...`) and inherits the `$(CinderRoot)` / `$(LuisaComputeRoot)` macros from the engine's `vc2022/NewTypeEngine.props`. To add Cinder blocks to the new project, run `cinder_blocks.py add ... --project D:/Projects/MyDemo` after this.
