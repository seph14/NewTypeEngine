#!/usr/bin/env python3
"""Cinder block manager for NewTypeEngine projects.

P1 scope (COMPLETE):
- Parse every cinderblock.xml under a Cinder blocks directory.
- Resolve MSW-targeted contributions (sources, headers, include paths, libs).
- Walk the <requires> dependency graph with cycle detection.
- `list` command: show installed (from sidecar JSON) and/or available blocks.
- `resolve` command: print the topological dependency closure of named blocks.

P2 scope (this revision):
- Marker-delimited region primitives for vcxproj/filters text.
- TextFile wrapper that preserves the original newline style (CRLF/LF).
- Vcxproj/Filters/Props editors that inject a one-time scaffold and reverse it.
- ProjectEditor orchestrator + round-trip tests proving byte-identity.

P3+ will add `add`, `remove`, `update` commands using these primitives.

Usage:
    python tools/cinder_blocks.py list --available
    python tools/cinder_blocks.py list --project D:/Projects/MyDemo
    python tools/cinder_blocks.py resolve TUIO
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default location of the Cinder blocks tree. Override with --blocks-dir or
# CINDER_BLOCKS_DIR env var. The sidecar JSON (P3+) records the effective
# directory per project so this default only matters for ad-hoc invocations.
#
# This MUST match the <CinderRoot> declared in vc2022/*.props: generated
# vcxproj paths anchor at $(CinderBlocksDir) = $(CinderRoot)\blocks, so a
# scan/reference mismatch here would produce paths that resolve to a
# different on-disk tree at build time.
DEFAULT_CINDER_BLOCKS_DIR = Path(r"../../Cinder/blocks")

# This engine is Windows-only; platform branches for other OSes are skipped.
TARGET_OS = "msw"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IncludePath:
    """An <includePath> entry. `cinder=true` means the path is relative to the
    Cinder root (not the block root); used by Cinder-WebSocketPP for asio.
    `absolute=true` emits the path verbatim — used by engine addons to reach
    in-tree files or MSBuild macros like `$(LCSInclude)`."""

    path: str
    system: bool = False
    cinder: bool = False
    absolute: bool = False


@dataclass
class StaticLib:
    """A <staticLibrary> entry. When `absolute=true`, the path is emitted
    verbatim (can be a linker flag like `-lz`, or a name resolved via lib
    search paths). `config` is set when the lib sits inside
    <platform config="debug"|"release">."""

    path: str
    absolute: bool = False
    config: str | None = None  # "debug" | "release" | None = both


@dataclass
class SourceFile:
    """A single source file. Comes from <source> (single) or <sourcePattern>
    (glob-expanded). `compile_as` mirrors the `compileAs` attribute used for
    Objective-C `.mm` files on Apple platforms (ignored on MSW).
    `absolute=true` emits the path verbatim — used by engine addons that
    reference in-tree sources (`..\\src\\newtype\\physics\\Physics.cpp`)."""

    path: str
    compile_as: str | None = None
    absolute: bool = False


@dataclass
class HeaderFile:
    """A single header file. Mirrors SourceFile; comes from <header> or
    <headerPattern>. `absolute=true` emits the path verbatim."""

    path: str
    absolute: bool = False


@dataclass
class Property:
    """A <property name="X">VALUE</property> declaration. Emitted as a new
    MSBuild `<X>VALUE</X>` line in the project's .props file. Used by engine
    addons to define machine-local roots (LCSRoot, etc.) without hard-coding
    them into the .props template."""

    name: str
    value: str


@dataclass
class BlockManifest:
    """Parsed and MSW-resolved view of one cinderblock.xml.

    `folder_name` is the canonical key used by this tool (it is the actual
    directory name and the path component used in generated vcxproj entries).
    `display_name` is the XML `name` attribute, kept for human display only.
    """

    folder_name: str
    display_name: str
    block_dir: Path
    block_id: str = ""
    summary: str = ""
    author: str = ""
    license: str = ""
    url: str = ""
    git: str = ""
    version: str = ""
    core: bool = False
    supported_os: set[str] = field(default_factory=set)  # empty = universal
    requires: list[str] = field(default_factory=list)  # block IDs

    # MSW-resolved contributions (non-MSW <platform> branches already excluded)
    include_paths: list[IncludePath] = field(default_factory=list)
    sources: list[SourceFile] = field(default_factory=list)
    headers: list[HeaderFile] = field(default_factory=list)
    static_libs: list[StaticLib] = field(default_factory=list)
    library_paths: list[str] = field(default_factory=list)
    build_copies: list[str] = field(default_factory=list)
    preprocessor_defines: list[str] = field(default_factory=list)
    additional_options: list[str] = field(default_factory=list)
    properties: list[Property] = field(default_factory=list)
    globals_properties: list[Property] = field(default_factory=list)
    copy_excludes: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)  # tracked, not used by add/remove

    @property
    def supports_msw(self) -> bool:
        return not self.supported_os or TARGET_OS in self.supported_os


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_manifest(block_dir: Path) -> BlockManifest:
    """Parse cinderblock.xml in `block_dir` and return the MSW-resolved manifest.

    Walks the XML once, descending only into <platform os="msw"> branches.
    Top-level elements apply to every platform including MSW. Non-MSW platforms
    are skipped entirely. Glob patterns (<sourcePattern>, <headerPattern>) are
    expanded against the filesystem at parse time.
    """
    xml_path = block_dir / "cinderblock.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"No cinderblock.xml in {block_dir}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    block_elem = root.find("block")
    if block_elem is None:
        raise ValueError(f"No <block> element in {xml_path}")

    m = BlockManifest(
        folder_name=block_dir.name,
        display_name=block_elem.get("name", block_dir.name),
        block_dir=block_dir,
        block_id=block_elem.get("id", ""),
        summary=block_elem.get("summary", ""),
        author=block_elem.get("author", ""),
        license=block_elem.get("license", ""),
        url=block_elem.get("url", ""),
        git=block_elem.get("git", ""),
        version=block_elem.get("version", ""),
        core=block_elem.get("core", "false").lower() == "true",
    )

    for tpl in root.findall("template"):
        if tpl.text:
            m.templates.append(tpl.text.strip())

    _walk_block(block_elem, m, config_filter=None)
    return m


def _walk_block(
    parent: ET.Element, m: BlockManifest, config_filter: str | None
) -> None:
    """Walk children of an XML element, populating the manifest.

    Only descends into <platform os="msw">. `config_filter` is set when inside
    <platform os="msw" config="debug"|"release"> so that <staticLibrary> entries
    can be tagged with their configuration.
    """
    for child in parent:
        # Comments and processing instructions appear as non-string tags.
        if not isinstance(child.tag, str):
            continue

        tag = child.tag
        text = (child.text or "").strip()

        if tag == "supports":
            os_val = child.get("os", "")
            if os_val:
                m.supported_os.add(os_val)

        elif tag == "requires":
            if text:
                m.requires.append(text)

        elif tag == "includePath":
            if text:
                m.include_paths.append(
                    IncludePath(
                        path=text,
                        system=child.get("system", "false").lower() == "true",
                        cinder=child.get("cinder", "false").lower() == "true",
                        absolute=child.get("absolute", "false").lower() == "true",
                    )
                )

        elif tag == "sourcePattern":
            for f in _expand_glob(m.block_dir, text):
                m.sources.append(SourceFile(path=f))

        elif tag == "headerPattern":
            for f in _expand_glob(m.block_dir, text):
                m.headers.append(HeaderFile(path=f))

        elif tag == "header":
            if text:
                m.headers.append(
                    HeaderFile(
                        path=text,
                        absolute=child.get("absolute", "false").lower() == "true",
                    )
                )

        elif tag == "source":
            if text:
                m.sources.append(
                    SourceFile(
                        path=text,
                        compile_as=child.get("compileAs"),
                        absolute=child.get("absolute", "false").lower() == "true",
                    )
                )

        elif tag == "staticLibrary":
            if text:
                m.static_libs.append(
                    StaticLib(
                        path=text,
                        absolute=child.get("absolute", "false").lower() == "true",
                        config=config_filter,
                    )
                )

        elif tag == "libraryPath":
            if text:
                m.library_paths.append(text)

        elif tag == "buildCopy":
            if text:
                m.build_copies.append(text)

        elif tag == "preprocessorDefine":
            if text:
                m.preprocessor_defines.append(text)

        elif tag == "additionalOption":
            # <additionalOption>VALUE</additionalOption> — appends VALUE
            # (e.g. `/bigobj`) to <AdditionalOptions> in every config.
            # Space-separated at emission time, matching MSBuild conventions.
            if text:
                m.additional_options.append(text)

        elif tag == "property":
            # <property name="X">VALUE</property> — emits <X>VALUE</X> in .props.
            # name is required; value may be empty (e.g. for property-driven
            # conditionals). Properties are only recognised at the top level /
            # <platform> scope of cinderblock.xml; nested walks preserve them
            # regardless of OS branch (they describe MSBuild state, not code).
            name = child.get("name")
            if name:
                m.properties.append(Property(name=name, value=text))

        elif tag == "globalsProperty":
            # <globalsProperty name="X">VALUE</globalsProperty> — emits
            # <X>VALUE</X> into the vcxproj's <PropertyGroup Label="Globals">.
            # Used for project-level MSBuild flags like <CppWinRTEnabled>true</CppWinRTEnabled>
            # that drive build-time toolchain selection.
            name = child.get("name")
            if name:
                m.globals_properties.append(Property(name=name, value=text))

        elif tag == "copyExclude":
            if text:
                m.copy_excludes.append(text)

        elif tag == "framework":
            # Apple-only; silently ignored on MSW.
            pass

        elif tag == "platform":
            os_val = child.get("os", "")
            config_val = (child.get("config") or "").lower() or None
            # Process if no os is specified (applies to all platforms, e.g.
            # Cinder-URG's top-level <platform config="debug">) or if os
            # matches our target. Skip only when os explicitly names another
            # platform (macosx, ios, ...).
            if os_val == "" or os_val == TARGET_OS:
                # Inside an MSW branch: keep walking, propagating any config.
                # If both outer and inner specify config, inner wins (matches
                # MSBuild's most-specific-wins semantics).
                _walk_block(child, m, config_filter=config_val or config_filter)
            # Non-MSW platform branch: skip entirely.

        # Unknown tags: skip silently. The cinderblock.xml schema is loose and
        # blocks may carry engine-specific extensions we don't yet recognise.


def _expand_glob(block_dir: Path, pattern: str) -> list[str]:
    """Expand a cinderblock.xml glob relative to `block_dir`.

    Returns sorted POSIX-style paths relative to `block_dir`. Missing
    directories yield an empty list (block may be partially shipped).
    """
    if not pattern:
        return []
    try:
        files = sorted(block_dir.glob(pattern))
    except (OSError, ValueError):
        return []
    return [f.relative_to(block_dir).as_posix() for f in files if f.is_file()]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class BlockRegistry:
    """Indexed view of every block under one or more blocks directories.

    All lookups are case-insensitive — block ids in the wild are inconsistent
    (e.g. OSC declares `org.libcinder.OSC` while TUIO requires `org.libcinder.osc`),
    and Cinder itself treats identifiers case-insensitively. Display/folder
    names follow the same rule for ergonomics.

    The registry walks every supplied root directory in order. Engine addon
    roots (e.g. `<engine>/engine_addons/`) are purely additive — they let
    `add sim` resolve alongside `add OSC` through the same lookup, and each
    BlockManifest keeps its own `block_dir` so per-block path anchoring stays
    correct without any mode-flag branching.
    """

    def __init__(
        self,
        blocks_dir: Path,
        extra_dirs: list[Path] | None = None,
    ):
        self.blocks_dir = Path(blocks_dir)
        self.extra_dirs: list[Path] = [Path(d) for d in (extra_dirs or [])]
        self._by_folder: dict[str, BlockManifest] = {}
        self._by_id: dict[str, BlockManifest] = {}
        self._errors: dict[str, str] = {}

    def scan(self) -> "BlockRegistry":
        """Parse every cinderblock.xml under every configured root. Idempotent."""
        self._by_folder.clear()
        self._by_id.clear()
        self._errors.clear()

        # The primary Cinder blocks dir is required; missing extras are
        # silently skipped (engine_addons/ may not exist on a fresh clone).
        if not self.blocks_dir.is_dir():
            raise FileNotFoundError(f"Blocks directory not found: {self.blocks_dir}")

        roots = [self.blocks_dir] + [
            d for d in self.extra_dirs if d.is_dir()
        ]
        for root in roots:
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                if not (entry / "cinderblock.xml").exists():
                    continue
                try:
                    m = parse_manifest(entry)
                    # First-seen wins for folder-name collisions so the
                    # primary blocks dir takes precedence over engine_addons.
                    if m.folder_name.lower() in self._by_folder:
                        continue
                    self._by_folder[m.folder_name.lower()] = m
                    if m.block_id:
                        key = m.block_id.lower()
                        if key in self._by_id:
                            self._errors[m.folder_name] = (
                                f"Duplicate block id '{m.block_id}' "
                                f"(also claimed by {self._by_id[key].folder_name})"
                            )
                        else:
                            self._by_id[key] = m
                except Exception as e:  # parse error on one block shouldn't kill the scan
                    self._errors[entry.name] = f"{type(e).__name__}: {e}"

        return self

    @property
    def blocks(self) -> dict[str, BlockManifest]:
        """Map of folder name -> manifest, keyed by original case."""
        return {m.folder_name: m for m in self._by_folder.values()}

    @property
    def errors(self) -> dict[str, str]:
        return self._errors

    def find(self, name_or_id: str) -> BlockManifest | None:
        """Look up by folder name, block id, or display name (case-insensitive)."""
        key = name_or_id.lower()
        if key in self._by_folder:
            return self._by_folder[key]
        if key in self._by_id:
            return self._by_id[key]
        for m in self._by_folder.values():
            if m.display_name.lower() == key:
                return m
        return None


# ---------------------------------------------------------------------------
# Dependency resolver
# ---------------------------------------------------------------------------


class DependencyCycleError(Exception):
    pass


class UnknownBlockError(Exception):
    pass


def resolve_deps(
    block_names: list[str],
    registry: BlockRegistry,
) -> list[tuple[str, str]]:
    """Resolve the transitive dependency closure of `block_names`.

    Returns a topologically sorted list of (folder_name, origin) tuples where
    origin is `"requested"` for inputs and `"dependency:<requiring_folder>"`
    for transitive deps. Dependencies appear before dependents.

    Raises UnknownBlockError if a name or required id cannot be located, and
    DependencyCycleError if the requires graph contains a cycle.
    """
    visited: set[str] = set()
    result: list[tuple[str, str]] = []
    stack: list[str] = []  # current DFS path, for cycle detection

    def visit(name: str, origin: str) -> None:
        m = registry.find(name)
        if m is None:
            raise UnknownBlockError(f"Block not found: {name}")
        folder = m.folder_name

        if folder in visited:
            return
        if folder in stack:
            cycle = " -> ".join(stack + [folder])
            raise DependencyCycleError(f"Dependency cycle: {cycle}")

        stack.append(folder)
        for dep_id in m.requires:
            dep = registry.find(dep_id)
            if dep is None:
                raise UnknownBlockError(
                    f"Block '{folder}' requires unknown block id '{dep_id}'"
                )
            visit(dep.folder_name, f"dependency:{folder}")
        stack.pop()

        visited.add(folder)
        result.append((folder, origin))

    for name in block_names:
        visit(name, "requested")

    return result


# ---------------------------------------------------------------------------
# Sidecar JSON (P1: read-only)
# ---------------------------------------------------------------------------


SIDECAR_FILENAME = ".cinder-blocks.json"


def sidecar_path(project_dir: Path) -> Path:
    return project_dir / "vc2022" / SIDECAR_FILENAME


def load_sidecar(project_dir: Path) -> dict | None:
    """Load the project's sidecar JSON, or return None if absent."""
    p = sidecar_path(project_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ===========================================================================
# P2: Project file editors (vcxproj / filters / props)
# ===========================================================================
#
# Strategy: text-based, not XML-tree-based. MSBuild vcxproj files have a
# finicky whitespace/line-ending style that ElementTree's serializer mangles.
# We instead locate stable anchors (existing XML comments, element tags) and
# splice our content in via regex + slicing. Everything outside our markers
# is preserved byte-for-byte.
#
# Each editor exposes `scaffold()` (one-time injection, idempotent) and
# `unscaffold()` (full reversal). Round-trip byte-identity is the P2 exit
# criterion: `scaffold(); unscaffold()` on a fresh generate_project.py output
# must yield the same bytes as the original.

# ---------------------------------------------------------------------------
# Region primitives
# ---------------------------------------------------------------------------

_REGION_PREFIX_BEGIN = "<!-- BEGIN cinder-blocks "
_REGION_PREFIX_END = "<!-- END cinder-blocks "
_REGION_SUFFIX = " -->"


def begin_marker(region: str) -> str:
    return f"{_REGION_PREFIX_BEGIN}{region}{_REGION_SUFFIX}"


def end_marker(region: str) -> str:
    return f"{_REGION_PREFIX_END}{region}{_REGION_SUFFIX}"


def _region_regex(region: str) -> re.Pattern[str]:
    """Regex matching a region block, capturing its inner content.

    A region is always laid out as:
        <leading newline><BEGIN marker><separator newline>
        <content>
        <END marker>
    The leading newline belongs to the region (so removal of an inserted
    region restores the pre-insertion byte stream exactly); the newline
    *after* the END marker belongs to the surrounding text. This asymmetry
    means round-trip is byte-exact: an ensure() that splits `X\\nY` into
    `X\\n<BEGIN>...<END>\\nY` is reversed by removing the matched span
    (`\\n<BEGIN>...<END>`), leaving `X\\nY`.
    """
    pattern = (
        r"(?P<lead>\r?\n)"
        + re.escape(begin_marker(region))
        + r"(?P<sep>\r?\n)"
        + r"(?P<content>.*?)"
        + re.escape(end_marker(region))
    )
    return re.compile(pattern, re.DOTALL)


def find_region(text: str, region: str) -> re.Match[str] | None:
    """Return the regex match for a region, or None if absent."""
    return _region_regex(region).search(text)


def ensure_region(
    text: str,
    region: str,
    anchor: str,
    content: str = "",
    place_after: bool = True,
    anchor_last_before: str | None = None,
) -> str:
    """Ensure a region exists. Insert it adjacent to `anchor` if absent.

    The region block is always `<nl>BEGIN...<END>` (no trailing newline) and
    is inserted either immediately after `anchor` (place_after=True, the
    default) or immediately before it (place_after=False). Round-trip
    removal (which consumes only the leading newline of the matched span)
    then restores the original byte stream exactly.

    `anchor_last_before`, if given, changes anchor matching to find the
    LAST occurrence of `anchor` before the first occurrence of
    `anchor_last_before` (e.g. the last `</ItemGroup>` before `<Import>`).
    Used to stack item-group regions at the end of the project body without
    colliding with earlier `</ItemGroup>` siblings.

    If the region already exists, the text is returned unchanged.
    """
    if find_region(text, region) is not None:
        return text

    if anchor_last_before is not None:
        bound_idx = text.find(anchor_last_before)
        if bound_idx < 0:
            raise ValueError(f"Bounding anchor not found: {anchor_last_before!r}")
        idx = text.rfind(anchor, 0, bound_idx + len(anchor_last_before))
        if idx < 0:
            raise ValueError(
                f"Anchor {anchor!r} not found before {anchor_last_before!r}"
            )
    else:
        idx = text.find(anchor)
        if idx < 0:
            raise ValueError(f"Anchor not found: {anchor!r}")

    nl = "\r\n" if "\r\n" in text else "\n"
    block = nl + begin_marker(region) + nl + content + nl + end_marker(region)
    if place_after:
        return text[: idx + len(anchor)] + block + text[idx + len(anchor) :]
    return text[:idx] + block + text[idx:]


def replace_region_content(text: str, region: str, content: str) -> str:
    """Replace the inner content of a region. Region must already exist."""
    m = find_region(text, region)
    if m is None:
        raise ValueError(f"Region '{region}' not found")
    return (
        text[: m.start()]
        + m.group("lead")
        + begin_marker(region)
        + m.group("sep")
        + content
        + end_marker(region)
        + text[m.end() :]
    )


def get_region_content(text: str, region: str) -> str | None:
    """Return the inner content of a region, or None if absent."""
    m = find_region(text, region)
    if m is None:
        return None
    return m.group("content")


def remove_region(text: str, region: str) -> str:
    """Remove a region including its markers and surrounding newlines."""
    m = find_region(text, region)
    if m is None:
        return text
    return text[: m.start()] + text[m.end() :]


# ---------------------------------------------------------------------------
# Inline reference injection
# ---------------------------------------------------------------------------
#
# Used to append a sentinel-bearing macro reference to an existing element's
# text content. Reversal keys off the sentinel (the unique macro name).

_ITEM_DEF_OPEN_RE = re.compile(r'<ItemDefinitionGroup\s+Condition="([^"]*)">')


def _find_enclosing_item_def_group(text: str, pos: int) -> str | None:
    """Return the source of the <ItemDefinitionGroup ...>...</ItemDefinitionGroup>
    block enclosing `pos`, or None if `pos` is not inside one.

    Used to scope injections to a particular configuration (Debug/Release).
    """
    candidates = list(_ITEM_DEF_OPEN_RE.finditer(text, 0, pos))
    if not candidates:
        return None
    open_match = candidates[-1]
    close_idx = text.find("</ItemDefinitionGroup>", open_match.end())
    if close_idx < 0 or close_idx < pos:
        return None
    close_end = close_idx + len("</ItemDefinitionGroup>")
    return text[open_match.start() : close_end]


def inject_suffix_before_tag(
    text: str, tag: str, suffix: str, restrict_to: str | None = None
) -> str:
    """Append `suffix` to the inner text of every `<tag>...</tag>` element.

    Idempotent: elements already containing `suffix` are left alone. If
    `restrict_to` is given, only elements inside an enclosing
    `<ItemDefinitionGroup>` whose opening tag contains the substring
    `restrict_to` are modified. This makes it possible to inject config-
    specific macros (e.g. only into the Debug ItemDefinitionGroup).

    Returns the new text.
    """
    close_tag = f"</{tag}>"
    open_pat = re.compile(rf"<{re.escape(tag)}(?:\s[^>]*)?>")

    out_chunks: list[str] = []
    last = 0
    pos = 0
    while True:
        idx = text.find(close_tag, pos)
        if idx < 0:
            out_chunks.append(text[last:])
            break
        opens = list(open_pat.finditer(text, 0, idx))
        if not opens:
            pos = idx + len(close_tag)
            continue
        open_match = opens[-1]
        body = text[open_match.end() : idx]

        if suffix in body:
            # Already injected.
            pos = idx + len(close_tag)
            continue

        if restrict_to is not None:
            scope = _find_enclosing_item_def_group(text, open_match.start())
            if scope is None or restrict_to not in scope:
                pos = idx + len(close_tag)
                continue

        out_chunks.append(text[last:idx])
        out_chunks.append(suffix)
        out_chunks.append(close_tag)
        last = idx + len(close_tag)
        pos = last
    return "".join(out_chunks)


def uninject_suffix_before_tag(
    text: str, tag: str, suffix: str
) -> str:
    """Remove `suffix` from inside every `<tag>...</tag>` element."""
    close_tag = f"</{tag}>"
    if suffix not in text:
        return text
    needle = suffix + close_tag
    return text.replace(needle, close_tag)


# ---------------------------------------------------------------------------
# TextFile: newline-preserving read/write
# ---------------------------------------------------------------------------


class TextFile:
    """A tiny text-file wrapper that preserves the original newline style.

    Internally, all manipulation works on `\n`-normalised text. On save, the
    newlines are rewritten to whatever style the file had on disk (defaulting
    to CRLF for new files). This makes round-trip byte-identity possible.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._text: str | None = None
        self._newline: str = "\r\n"  # default for newly-created files

    def load(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        data = self.path.read_bytes()
        if b"\r\n" in data:
            self._newline = "\r\n"
        else:
            self._newline = "\n"
        # Normalise to \n internally. Decode after preserving byte content.
        self._text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        return self._text

    def save(self) -> None:
        if self._text is None:
            raise RuntimeError(f"Nothing to save; load() {self.path} first")
        encoded = self._text.replace("\n", self._newline).encode("utf-8")
        self.path.write_bytes(encoded)

    @property
    def text(self) -> str:
        if self._text is None:
            return self.load()
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        # Normalise incoming text to \n; save() will re-encode.
        self._text = value.replace("\r\n", "\n").replace("\r", "\n")

    def is_modified(self) -> bool:
        """True if in-memory text differs from disk (or file is new)."""
        if not self.path.exists():
            return self._text is not None
        if self._text is None:
            return False
        return self.text != TextFile(self.path).load()


# ---------------------------------------------------------------------------
# VcxprojEditor
# ---------------------------------------------------------------------------

# Anchor strings used to position region insertions. These must appear
# verbatim in any vcxproj produced by tools/generate_project.py.
_ANCHOR_USER_MACROS = '<PropertyGroup Label="UserMacros" />'
_ANCHOR_TARGETS_IMPORT = '<Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />'
_ANCHOR_LAST_ITEMGROUP_CLOSE = "</ItemGroup>"
# Stable anchor inside <PropertyGroup Label="Globals">. Always emitted by
# tools/generate_project.py; safe to key off regardless of project name.
_ANCHOR_GLOBALS_KEYWORD = "<Keyword>Win32Proj</Keyword>"

# Region names.
REGION_PROPERTIES = "properties"
REGION_CLCOMPILE = "ClCompile"
REGION_CLINCLUDE = "ClInclude"
# Holds addon-emitted project-level MSBuild flags (e.g. <CppWinRTEnabled>true</CppWinRTEnabled>)
# inside <PropertyGroup Label="Globals">. Empty until a <globalsProperty> addon is installed.
REGION_VCXPROJ_GLOBALS = "vcxproj globals-properties"

# MSBuild macros defined inside the `properties` region. The empty-string
# defaults mean any reference expands to nothing, which MSBuild treats
# gracefully for `;`-separated lists. For PostBuild we use a chained
# property that is undefined when empty, so `<Command>... $(Undefined)</Command>`
# expands to `... ` rather than `... &` (which would break cmd.exe).
_SCAFFOLD_PROPERTY_GROUP = """<PropertyGroup Label="CinderBlocks">
    <CinderBlocksIncludePaths></CinderBlocksIncludePaths>
    <CinderBlocksLibPaths></CinderBlocksLibPaths>
    <CinderBlocksDebugLibs></CinderBlocksDebugLibs>
    <CinderBlocksReleaseLibs></CinderBlocksReleaseLibs>
    <CinderBlocksPostBuild></CinderBlocksPostBuild>
    <CinderBlocksPostBuildChained Condition="'$(CinderBlocksPostBuild)' != ''">&amp; $(CinderBlocksPostBuild)</CinderBlocksPostBuildChained>
    <CinderBlocksDefines></CinderBlocksDefines>
    <CinderBlocksAdditionalOptions></CinderBlocksAdditionalOptions>
  </PropertyGroup>"""

_SCAFFOLD_EMPTY_ITEMGROUP = "<ItemGroup>\n  </ItemGroup>"

# Macro references appended to existing element bodies. The leading
# separator (`;` or space) is part of the suffix so removal is exact.
_SUFFIX_INCLUDE_PATHS = ";$(CinderBlocksIncludePaths)"
_SUFFIX_DEBUG_LIBS = ";$(CinderBlocksDebugLibs)"
_SUFFIX_RELEASE_LIBS = ";$(CinderBlocksReleaseLibs)"
_SUFFIX_DEFINES = ";$(CinderBlocksDefines)"
_SUFFIX_ADDITIONAL_OPTIONS = " $(CinderBlocksAdditionalOptions)"
_SUFFIX_POSTBUILD = " $(CinderBlocksPostBuildChained)"

# All suffixes for batch (un)injection. PostBuild is config-agnostic; the
# lib suffixes are config-specific and applied via restrict_to windows.
# Preprocessor defines and additional options are config-agnostic.
_ALL_SUFFIX_INJECTIONS = [
    ("AdditionalIncludeDirectories", _SUFFIX_INCLUDE_PATHS, None),
    ("AdditionalDependencies", _SUFFIX_DEBUG_LIBS, "=='Debug|x64'"),
    ("AdditionalDependencies", _SUFFIX_DEBUG_LIBS, "=='Debug_Runtime|x64'"),
    ("AdditionalDependencies", _SUFFIX_RELEASE_LIBS, "=='Release|x64'"),
    ("PreprocessorDefinitions", _SUFFIX_DEFINES, None),
    ("AdditionalOptions", _SUFFIX_ADDITIONAL_OPTIONS, None),
    ("Command", _SUFFIX_POSTBUILD, None),
]


class VcxprojEditor:
    """Marker-based editor for a .vcxproj file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = TextFile(self.path)

    def load(self) -> "VcxprojEditor":
        self.file.load()
        return self

    def save(self) -> None:
        self.file.save()

    @property
    def text(self) -> str:
        return self.file.text

    @text.setter
    def text(self, value: str) -> None:
        self.file.text = value

    def is_scaffolded(self) -> bool:
        return find_region(self.text, REGION_PROPERTIES) is not None

    def scaffold(self) -> bool:
        """Inject markers and property references. Idempotent.

        Returns True if any change was made.
        """
        original = self.text
        text = original

        # 1. Insert the regions. The properties region sits near the top;
        #    ClCompile and ClInclude chain off each other at the bottom (just
        #    before the targets import) so they stay in deterministic order;
        #    the globals-properties region anchors inside the Globals
        #    PropertyGroup at the project's <Keyword> flag. All use
        #    place_after=True so newline accounting is uniform.
        text = ensure_region(
            text,
            REGION_PROPERTIES,
            _ANCHOR_USER_MACROS,
            content=_SCAFFOLD_PROPERTY_GROUP,
        )
        text = ensure_region(
            text,
            REGION_CLCOMPILE,
            _ANCHOR_LAST_ITEMGROUP_CLOSE,
            content=_SCAFFOLD_EMPTY_ITEMGROUP,
            anchor_last_before=_ANCHOR_TARGETS_IMPORT,
        )
        text = ensure_region(
            text,
            REGION_CLINCLUDE,
            end_marker(REGION_CLCOMPILE),
            content=_SCAFFOLD_EMPTY_ITEMGROUP,
        )
        text = ensure_region(
            text,
            REGION_VCXPROJ_GLOBALS,
            _ANCHOR_GLOBALS_KEYWORD,
            content="",
        )

        # 2. Inject `;$(...)` / ` $(...)` suffixes into existing element bodies.
        for tag, suffix, restrict in _ALL_SUFFIX_INJECTIONS:
            text = inject_suffix_before_tag(text, tag, suffix, restrict_to=restrict)

        if text != original:
            self.file.text = text
            return True
        return False

    def unscaffold(self) -> bool:
        """Reverse scaffold(). Returns True if any change was made."""
        original = self.text
        text = original

        for tag, suffix, _ in _ALL_SUFFIX_INJECTIONS:
            text = uninject_suffix_before_tag(text, tag, suffix)

        for region in (
            REGION_PROPERTIES,
            REGION_CLCOMPILE,
            REGION_CLINCLUDE,
            REGION_VCXPROJ_GLOBALS,
        ):
            text = remove_region(text, region)

        if text != original:
            self.file.text = text
            return True
        return False


# ---------------------------------------------------------------------------
# FiltersEditor
# ---------------------------------------------------------------------------

_ANCHOR_END_PROJECT_FILTERS = "</Project>"

REGION_FILTERS_FILTER = "filters Filter"
REGION_FILTERS_CLCOMPILE = "filters ClCompile"
REGION_FILTERS_CLINCLUDE = "filters ClInclude"

_SCAFFOLD_FILTERS_EMPTY_ITEMGROUP = "<ItemGroup>\n  </ItemGroup>"


class FiltersEditor:
    """Marker-based editor for a .vcxproj.filters file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = TextFile(self.path)

    def load(self) -> "FiltersEditor":
        self.file.load()
        return self

    def save(self) -> None:
        self.file.save()

    @property
    def text(self) -> str:
        return self.file.text

    @text.setter
    def text(self, value: str) -> None:
        self.file.text = value

    def is_scaffolded(self) -> bool:
        return find_region(self.text, REGION_FILTERS_FILTER) is not None

    def scaffold(self) -> bool:
        original = self.text
        text = original
        # Stack the three regions at the end of the filters file, chained
        # off each other so the order is deterministic. The first region
        # anchors on the last </ItemGroup> before </Project>; subsequent
        # regions chain off the previous region's END marker.
        text = ensure_region(
            text,
            REGION_FILTERS_FILTER,
            "</ItemGroup>",
            content=_SCAFFOLD_FILTERS_EMPTY_ITEMGROUP,
            anchor_last_before=_ANCHOR_END_PROJECT_FILTERS,
        )
        text = ensure_region(
            text,
            REGION_FILTERS_CLCOMPILE,
            end_marker(REGION_FILTERS_FILTER),
            content=_SCAFFOLD_FILTERS_EMPTY_ITEMGROUP,
        )
        text = ensure_region(
            text,
            REGION_FILTERS_CLINCLUDE,
            end_marker(REGION_FILTERS_CLCOMPILE),
            content=_SCAFFOLD_FILTERS_EMPTY_ITEMGROUP,
        )
        if text != original:
            self.file.text = text
            return True
        return False

    def unscaffold(self) -> bool:
        original = self.text
        text = original
        for region in (
            REGION_FILTERS_FILTER,
            REGION_FILTERS_CLCOMPILE,
            REGION_FILTERS_CLINCLUDE,
        ):
            text = remove_region(text, region)
        if text != original:
            self.file.text = text
            return True
        return False


# ---------------------------------------------------------------------------
# PropsEditor
# ---------------------------------------------------------------------------

# Sentinel-based injection of <CinderBlocksDir> and <CinderBlocksLocalDir>
# properties after the existing <CinderRoot>...</CinderRoot> element.
#   * CinderBlocksDir    — points at the shared block tree (reference mode).
#   * CinderBlocksLocalDir — points at the project's own copy (copy mode).
# Both are always emitted; unused macros cost nothing and avoiding
# conditional injection keeps the scaffold/unscaffold round-trip clean.
_PROPS_CINDERROOT_CLOSE = "</CinderRoot>"
_PROPS_CINDER_BLOCKS_LINES = (
    "\n    <CinderBlocksDir>$(CinderRoot)\\blocks</CinderBlocksDir>"
    "\n    <CinderBlocksLocalDir>..\\blocks</CinderBlocksLocalDir>"
)
_PROPS_CINDERBLOCKSDIR_TAG = "<CinderBlocksDir>"
_PROPS_CINDERBLOCKSLOCALDIR_TAG = "<CinderBlocksLocalDir>"
_PROPS_CINDERBLOCKSLOCALDIR_CLOSE = "</CinderBlocksLocalDir>"

# Marker-delimited region for addon-emitted MSBuild properties (one
# `<X>VALUE</X>` line per `<property>` declaration in cinderblock.xml).
# Anchored on the closing tag of <CinderBlocksLocalDir> so it always sits
# after the two fixed scaffold lines and before whatever follows in the
# engine's .props template. Empty when no addon contributes properties.
REGION_PROPS_PROPERTIES = "props properties"


class PropsEditor:
    """Marker-based editor for a .props file.

    Manages two layers:
      1. Always-on injected lines: `<CinderBlocksDir>`, `<CinderBlocksLocalDir>`.
      2. Marker region `props properties` for addon-emitted properties.
    Both layers are inserted by `scaffold()` and removed by `unscaffold()`.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = TextFile(self.path)

    def load(self) -> "PropsEditor":
        self.file.load()
        return self

    def save(self) -> None:
        self.file.save()

    @property
    def text(self) -> str:
        return self.file.text

    @text.setter
    def text(self, value: str) -> None:
        self.file.text = value

    def is_scaffolded(self) -> bool:
        return _PROPS_CINDERBLOCKSDIR_TAG in self.text

    def scaffold(self) -> bool:
        text = self.text
        if _PROPS_CINDERBLOCKSDIR_TAG in text:
            # Already has the fixed lines. Ensure the region exists too —
            # older scaffolds (pre-properties-region) won't have it.
            new_text = self._ensure_properties_region(text)
            if new_text != text:
                self.file.text = new_text
                return True
            return False
        idx = text.find(_PROPS_CINDERROOT_CLOSE)
        if idx < 0:
            raise ValueError(
                f"{self.path}: could not find {_PROPS_CINDERROOT_CLOSE!r} anchor"
            )
        nl = "\r\n" if "\r\n" in text else "\n"
        insertion = _PROPS_CINDER_BLOCKS_LINES.replace("\n", nl)
        insert_at = idx + len(_PROPS_CINDERROOT_CLOSE)
        text = text[:insert_at] + insertion + text[insert_at:]
        # Now ensure the properties region is in place, anchored on the
        # freshly-inserted </CinderBlocksLocalDir> close tag.
        text = self._ensure_properties_region(text)
        self.file.text = text
        return True

    def _ensure_properties_region(self, text: str) -> str:
        """Insert the empty properties region if absent. Idempotent."""
        return ensure_region(
            text,
            REGION_PROPS_PROPERTIES,
            _PROPS_CINDERBLOCKSLOCALDIR_CLOSE,
            content="",
        )

    def unscaffold(self) -> bool:
        text = self.text
        if _PROPS_CINDERBLOCKSDIR_TAG not in text:
            return False
        # Remove the properties region first (anchored on a tag we're about
        # to delete); then strip the two fixed lines.
        text = remove_region(text, REGION_PROPS_PROPERTIES)
        nl = "\r\n" if "\r\n" in text else "\n"
        for tag in (_PROPS_CINDERBLOCKSDIR_TAG, _PROPS_CINDERBLOCKSLOCALDIR_TAG):
            while True:
                tag_idx = text.find(tag)
                if tag_idx < 0:
                    break
                line_start = text.rfind(nl, 0, tag_idx)
                if line_start < 0:
                    line_start = 0
                else:
                    line_start += len(nl)
                line_end = text.find(nl, tag_idx)
                if line_end < 0:
                    line_end = len(text)
                else:
                    line_end += len(nl)
                text = text[:line_start] + text[line_end:]
        if text != self.text:
            self.file.text = text
            return True
        return False


# ---------------------------------------------------------------------------
# ProjectEditor: orchestrator across vcxproj/filters/props
# ---------------------------------------------------------------------------


class ProjectEditor:
    """Coordinates Vcxproj/Filters/Props editors for one project.

    Construct with the path to the .vcxproj file; sibling files are located
    by extension.
    """

    def __init__(self, vcxproj_path: Path):
        self.vcxproj_path = Path(vcxproj_path)
        self.filters_path = self.vcxproj_path.with_suffix(".vcxproj.filters")
        self.props_path = self.vcxproj_path.with_suffix(".props")
        self.vcxproj = VcxprojEditor(self.vcxproj_path)
        self.filters = FiltersEditor(self.filters_path)
        self.props = PropsEditor(self.props_path)

    def load(self) -> "ProjectEditor":
        self.vcxproj.load()
        if self.filters_path.exists():
            self.filters.load()
        if self.props_path.exists():
            self.props.load()
        return self

    def save(self) -> None:
        self.vcxproj.save()
        if self.filters_path.exists():
            self.filters.save()
        if self.props_path.exists():
            self.props.save()

    def is_scaffolded(self) -> bool:
        return self.vcxproj.is_scaffolded()

    def scaffold(self) -> bool:
        changed = False
        if not self.filters_path.exists():
            raise FileNotFoundError(f"Missing filters file: {self.filters_path}")
        if not self.props_path.exists():
            raise FileNotFoundError(f"Missing props file: {self.props_path}")
        # Ensure all editors have loaded text.
        self.load()
        if self.vcxproj.scaffold():
            changed = True
        if self.filters.scaffold():
            changed = True
        if self.props.scaffold():
            changed = True
        if changed:
            self.save()
        return changed

    def unscaffold(self) -> bool:
        changed = False
        self.load()
        if self.vcxproj.unscaffold():
            changed = True
        if self.filters_path.exists() and self.filters.unscaffold():
            changed = True
        if self.props_path.exists() and self.props.unscaffold():
            changed = True
        if changed:
            self.save()
        return changed


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _resolve_blocks_dir(arg: str | None) -> Path:
    """Pick the blocks directory from arg, env, or default (in that order)."""
    import os

    if arg:
        return Path(arg)
    env = os.environ.get("CINDER_BLOCKS_DIR")
    if env:
        return Path(env)
    return DEFAULT_CINDER_BLOCKS_DIR


# Engine addons live alongside the engine sources (typically at
# `<engine>/engine_addons/`). Override per-invocation with --engine-addons-dir
# or set NT_ENGINE_ADDONS_DIR in the environment. An absent directory is
# silently skipped by the registry scan.
_ENGINE_ADDONS_SUBDIR = "engine_addons"


def _default_engine_addons_dir() -> Path | None:
    """Return `<engine>/engine_addons/` if this script lives inside the engine
    tree, else None. tools/ is two levels below the engine root."""
    import os

    tools_dir = Path(__file__).resolve().parent
    candidate = tools_dir.parent / _ENGINE_ADDONS_SUBDIR
    return candidate if candidate.is_dir() else None


def _resolve_engine_addons_dir(arg: str | None) -> Path | None:
    """Pick the engine addons directory from arg, env, or default location."""
    import os

    if arg:
        return Path(arg)
    env = os.environ.get("NT_ENGINE_ADDONS_DIR")
    if env:
        return Path(env)
    return _default_engine_addons_dir()


def _build_registry(args) -> BlockRegistry:
    """Construct a BlockRegistry covering both the Cinder blocks dir and the
    engine addons dir derived from `args`. Missing engine addons dir is OK.
    """
    blocks_dir = _resolve_blocks_dir(getattr(args, "blocks_dir", None))
    extras: list[Path] = []
    ea = _resolve_engine_addons_dir(getattr(args, "engine_addons_dir", None))
    if ea is not None:
        extras.append(ea)
    return BlockRegistry(blocks_dir, extra_dirs=extras).scan()


# ===========================================================================
# P3: Block-application layer (sidecar JSON, macro builder, region content)
# ===========================================================================


SIDECAR_SCHEMA_VERSION = 1


def init_sidecar(blocks_dir: Path) -> dict:
    """Return an empty sidecar dict pointing at the given blocks directory."""
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "cinder_blocks_dir": str(blocks_dir),
        "blocks": {},
    }


def save_sidecar(project_dir: Path, data: dict) -> Path:
    """Write the sidecar JSON. Returns the path written.

    Formatting is deterministic so the file is diff-friendly in version
    control: UTF-8, LF newlines, 2-space indent, sorted block keys.
    """
    p = sidecar_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    p.write_bytes(text.encode("utf-8"))
    return p


# ---------------------------------------------------------------------------
# Path strategy
# ---------------------------------------------------------------------------
#
# Two install modes:
#   * reference — every block contribution is emitted as a path anchored at
#     `$(CinderBlocksDir)`, which is itself defined in <project>.props as
#     `$(CinderRoot)\blocks`. The user can repoint the entire block tree by
#     editing a single .props line; nothing is copied into the project.
#   * copy — block files are duplicated into `<project>/blocks/<folder>/`
#     and paths anchor at `$(CinderBlocksLocalDir)` (defined as `..\blocks`
#     relative to the vc2022/ project directory). Self-contained projects
#     that travel with their block sources; trades disk for portability.
#
# Special cases:
#   * <includePath cinder="true"> -> path is relative to $(CinderRoot), not
#     the block root. Used by Cinder-WebSocketPP to reach Cinder's bundled
#     asio headers. Mode-agnostic (Cinder tree is always external).
#   * <staticLibrary absolute="true"> -> emit verbatim. The value can be a
#     bare lib name (e.g. `libpng.lib` resolved via lib search paths) or a
#     linker flag (`-lz`).
#   * Windows backslashes in generated paths, matching MSBuild conventions.


# Mode constants — strings kept narrow for serialisation into sidecar JSON.
MODE_REFERENCE = "reference"
MODE_COPY = "copy"


def _block_path_rel(folder: str, path: str, mode: str) -> str:
    """Format a block-rooted path for vcxproj.

    Reference mode anchors at `$(CinderBlocksDir)`; copy mode anchors at
    `$(CinderBlocksLocalDir)`.
    """
    anchor = (
        r"$(CinderBlocksLocalDir)"
        if mode == MODE_COPY
        else r"$(CinderBlocksDir)"
    )
    parts = [anchor, folder]
    parts.extend(p for p in path.replace("/", "\\").split("\\") if p)
    return "\\".join(parts)


def _cinder_path_rel(path: str) -> str:
    """Format a Cinder-rooted path (`cinder=true` includePath)."""
    parts = [r"$(CinderRoot)"]
    parts.extend(p for p in path.replace("/", "\\").split("\\") if p)
    return "\\".join(parts)


@dataclass
class BlockInstall:
    """A block plus the mode it was installed in."""

    manifest: BlockManifest
    mode: str = MODE_REFERENCE


# ---------------------------------------------------------------------------
# Filter GUID helpers
# ---------------------------------------------------------------------------
#
# Each block contributes a `Blocks\<folder>` filter plus subfilters per
# source/include subdirectory it ships. GUIDs are deterministic uuid5 from
# the filter's full name so the same block always lands under the same
# virtual folder across runs (matters for VS's "open project" caching).

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "newtype.cinder-blocks")


def _filter_guid(filter_name: str) -> str:
    return "{" + str(uuid.uuid5(_NAMESPACE, f"filter.{filter_name}")) + "}"


def _filter_name_for_block(folder: str, sub: str | None = None) -> str:
    """Return the full filter path for a block (and optional subdir)."""
    if sub:
        # sub is a POSIX-style relative path; take its dirname when the file
        # sits in a nested folder, or empty if at the block root.
        parts = [p for p in sub.replace("\\", "/").split("/") if p]
        if len(parts) > 1:
            sub_parts = parts[:-1]
            return f"Blocks\\{folder}\\" + "\\".join(sub_parts)
        return f"Blocks\\{folder}"
    return f"Blocks\\{folder}"


def _filter_for_absolute(path: str, kind_filter: str) -> str:
    """Derive a filter name from an absolute path (engine addons).

    For in-tree engine sources like `..\\src\\newtype\\physics\\Physics.cpp`,
    the result is `Source Files\\newtype\\physics`. Leading `..` segments and
    a `src`/`include` first-non-dot segment are stripped before joining; the
    file's own directory becomes the deepest filter. Paths that don't match
    this shape still produce a sensible nested filter.
    """
    parts = [p for p in path.replace("\\", "/").split("/") if p and p != "."]
    while parts and parts[0] == "..":
        parts = parts[1:]
    if parts and parts[0] in ("src", "include"):
        parts = parts[1:]
    if len(parts) > 1:
        return kind_filter + "\\" + "\\".join(parts[:-1])
    return kind_filter


def _emit_src_path(src: "SourceFile", folder: str, mode: str) -> str:
    """Resolve a SourceFile to its emitted vcxproj path string."""
    if src.absolute:
        return src.path.replace("/", "\\")
    return _block_path_rel(folder, src.path, mode)


def _emit_hdr_path(hdr: "HeaderFile", folder: str, mode: str) -> str:
    """Resolve a HeaderFile to its emitted vcxproj path string."""
    if hdr.absolute:
        return hdr.path.replace("/", "\\")
    return _block_path_rel(folder, hdr.path, mode)


def _filter_for_src(src: "SourceFile", folder: str) -> str:
    """Filter name for a source file. Absolute paths map to Source Files\\..."""
    if src.absolute:
        return _filter_for_absolute(src.path, "Source Files")
    return _filter_name_for_block(folder, src.path)


def _filter_for_hdr(hdr: "HeaderFile", folder: str) -> str:
    """Filter name for a header file."""
    if hdr.absolute:
        return _filter_for_absolute(hdr.path, "Header Files")
    return _filter_name_for_block(folder, hdr.path)


# ---------------------------------------------------------------------------
# MacroBuilder
# ---------------------------------------------------------------------------


@dataclass
class MacroValues:
    """The scaffold MSBuild property values, derived from manifests.

    `post_build` is the raw chained command body (without the leading
    `& `); the scaffold's `CinderBlocksPostBuildChained` property wraps it.
    `defines` carries preprocessor defines joined by `;`, emitted via
    `$(CinderBlocksDefines)` into `<PreprocessorDefinitions>` in every config.
    `additional_options` carries compile flags joined by space, emitted via
    `$(CinderBlocksAdditionalOptions)` into `<AdditionalOptions>` in every config.
    `props_lines` carries `<X>VALUE</X>` lines (one per manifest `<property>`)
    that get emitted into a region in the .props file.
    `globals_lines` carries the same shape but targets a region inside the
    vcxproj's `<PropertyGroup Label="Globals">` (project-level flags like
    `<CppWinRTEnabled>true</CppWinRTEnabled>`).
    """

    include_paths: str = ""
    lib_paths: str = ""
    debug_libs: str = ""
    release_libs: str = ""
    post_build: str = ""
    defines: str = ""
    additional_options: str = ""
    props_lines: str = ""
    globals_lines: str = ""


def build_macros(installs: list[BlockInstall]) -> MacroValues:
    """Compute macro values from a list of (manifest, mode) installs."""
    inc: list[str] = []
    lib_paths: list[str] = []
    debug_libs: list[str] = []
    release_libs: list[str] = []
    post_build: list[str] = []
    defines: list[str] = []
    additional_options: list[str] = []
    props: list[str] = []
    globals_lines: list[str] = []

    for inst in installs:
        m = inst.manifest
        for ip in m.include_paths:
            if ip.absolute:
                inc.append(ip.path.replace("/", "\\"))
            elif ip.cinder:
                inc.append(_cinder_path_rel(ip.path))
            else:
                inc.append(_block_path_rel(m.folder_name, ip.path, inst.mode))
        for lp in m.library_paths:
            lib_paths.append(_block_path_rel(m.folder_name, lp, inst.mode))
        for lib in m.static_libs:
            path = (
                lib.path
                if lib.absolute
                else _block_path_rel(m.folder_name, lib.path, inst.mode)
            )
            cfg = (lib.config or "").lower()
            if cfg == "release":
                release_libs.append(path)
            elif cfg == "debug":
                debug_libs.append(path)
            else:
                # Universal: emit into both pipelines.
                debug_libs.append(path)
                release_libs.append(path)
        for bc in m.build_copies:
            src = _block_path_rel(m.folder_name, bc, inst.mode)
            post_build.append(
                f'xcopy /y "{src}" "$(OutDir)\\"'
            )
        for define in m.preprocessor_defines:
            defines.append(define)
        for opt in m.additional_options:
            additional_options.append(opt)
        for prop in m.properties:
            # Emit `    <NAME>VALUE</NAME>` — the per-line indentation matches
            # the surrounding .props style so the file stays diff-friendly.
            props.append(f"    <{prop.name}>{prop.value}</{prop.name}>")
        for prop in m.globals_properties:
            globals_lines.append(f"  <{prop.name}>{prop.value}</{prop.name}>")

    return MacroValues(
        include_paths=";".join(inc),
        lib_paths=";".join(lib_paths),
        debug_libs=";".join(debug_libs),
        release_libs=";".join(release_libs),
        post_build=" &amp; ".join(post_build),
        defines=";".join(defines),
        additional_options=" ".join(additional_options),
        props_lines="\n".join(props),
        globals_lines="\n".join(globals_lines),
    )


# ---------------------------------------------------------------------------
# Region content builders
# ---------------------------------------------------------------------------


def _indent(text: str, level: int) -> str:
    """Indent every non-empty line by `level` pairs of 2 spaces."""
    pad = "  " * level
    return "\n".join(
        (pad + line if line else line) for line in text.splitlines()
    )


def build_vcxproj_clcompile(installs: list[BlockInstall]) -> str:
    """Inner content for the vcxproj ClCompile region."""
    if not installs:
        return "  <ItemGroup>\n  </ItemGroup>"
    lines = ["  <ItemGroup>"]
    for inst in installs:
        for src in inst.manifest.sources:
            lines.append(
                f'    <ClCompile Include="{_emit_src_path(src, inst.manifest.folder_name, inst.mode)}" />'
            )
    lines.append("  </ItemGroup>")
    return "\n".join(lines)


def build_vcxproj_clinclude(installs: list[BlockInstall]) -> str:
    """Inner content for the vcxproj ClInclude region."""
    if not installs:
        return "  <ItemGroup>\n  </ItemGroup>"
    lines = ["  <ItemGroup>"]
    for inst in installs:
        for hdr in inst.manifest.headers:
            lines.append(
                f'    <ClInclude Include="{_emit_hdr_path(hdr, inst.manifest.folder_name, inst.mode)}" />'
            )
    lines.append("  </ItemGroup>")
    return "\n".join(lines)


def _filters_filter_definitions(installs: list[BlockInstall]) -> list[str]:
    """Return ordered unique filter names (with full parent chain) for the blocks.

    Includes the top-level `Blocks` filter so it appears alongside the
    project's `Source Files` / `Header Files` / `Resource Files` filters
    in Solution Explorer, matching TinderBox's layout. Engine addons with
    absolute paths produce filters under `Source Files`/`Header Files`
    directly (no `Blocks\\<folder>` prefix).
    """
    seen: dict[str, None] = {}
    # Always include the top-level Blocks filter when any *non-absolute*
    # install contributes a filter (i.e. a real Cinder block). Absolute-only
    # addons don't need the Blocks root.
    has_block_install = any(
        any(not s.absolute for s in inst.manifest.sources)
        or any(not h.absolute for h in inst.manifest.headers)
        for inst in installs
    )
    if has_block_install:
        seen["Blocks"] = None
    for inst in installs:
        m = inst.manifest
        names_for_block: dict[str, None] = {}
        if any(not s.absolute for s in m.sources) or any(not h.absolute for h in m.headers):
            names_for_block[_filter_name_for_block(m.folder_name)] = None
        for src in m.sources:
            names_for_block[_filter_for_src(src, m.folder_name)] = None
        for hdr in m.headers:
            names_for_block[_filter_for_hdr(hdr, m.folder_name)] = None
        # Walk the entire parent chain so intermediate filters are also defined.
        for name in list(names_for_block):
            parts = name.split("\\")
            for i in range(1, len(parts) + 1):
                parent = "\\".join(parts[:i])
                seen.setdefault(parent, None)
    return list(seen.keys())


def build_filters_filter(installs: list[BlockInstall]) -> str:
    """Inner content for the .filters Filter-definitions region."""
    if not installs:
        return "  <ItemGroup>\n  </ItemGroup>"
    lines = ["  <ItemGroup>"]
    for name in _filters_filter_definitions(installs):
        guid = _filter_guid(name)
        lines.append(f'    <Filter Include="{name}">')
        lines.append(f"      <UniqueIdentifier>{guid}</UniqueIdentifier>")
        lines.append("    </Filter>")
    lines.append("  </ItemGroup>")
    return "\n".join(lines)


def build_filters_clcompile(installs: list[BlockInstall]) -> str:
    """Inner content for the .filters ClCompile region."""
    if not installs:
        return "  <ItemGroup>\n  </ItemGroup>"
    lines = ["  <ItemGroup>"]
    for inst in installs:
        m = inst.manifest
        for src in m.sources:
            filt = _filter_for_src(src, m.folder_name)
            path = _emit_src_path(src, m.folder_name, inst.mode)
            lines.append(f'    <ClCompile Include="{path}">')
            lines.append(f"      <Filter>{filt}</Filter>")
            lines.append("    </ClCompile>")
    lines.append("  </ItemGroup>")
    return "\n".join(lines)


def build_filters_clinclude(installs: list[BlockInstall]) -> str:
    """Inner content for the .filters ClInclude region."""
    if not installs:
        return "  <ItemGroup>\n  </ItemGroup>"
    lines = ["  <ItemGroup>"]
    for inst in installs:
        m = inst.manifest
        for hdr in m.headers:
            filt = _filter_for_hdr(hdr, m.folder_name)
            path = _emit_hdr_path(hdr, m.folder_name, inst.mode)
            lines.append(f'    <ClInclude Include="{path}">')
            lines.append(f"      <Filter>{filt}</Filter>")
            lines.append("    </ClInclude>")
    lines.append("  </ItemGroup>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ProjectState: ties sidecar + manifests + editors together
# ---------------------------------------------------------------------------


@dataclass
class _InstalledBlock:
    """A block listed in the sidecar, augmented with its parsed manifest."""

    info: dict  # raw sidecar entry
    manifest: BlockManifest


class ProjectState:
    """In-memory view of a scaffolded project: sidecar + parsed manifests.

    Used by cmd_add/cmd_remove/cmd_update. Handles loading the sidecar,
    parsing each installed block's manifest from the recorded blocks dir,
    and writing back both the sidecar and the project files.
    """

    def __init__(self, project_dir: Path, registry: BlockRegistry):
        self.project_dir = Path(project_dir).resolve()
        self.registry = registry
        self.vcxproj_path = self.project_dir / "vc2022" / (
            self.project_dir.name + ".vcxproj"
        )
        self._sidecar: dict | None = None
        self._manifests: dict[str, BlockManifest] = {}

    @property
    def sidecar_path(self) -> Path:
        return sidecar_path(self.project_dir)

    def load(self) -> "ProjectState":
        data = load_sidecar(self.project_dir)
        if data is None:
            raise FileNotFoundError(
                f"No sidecar JSON at {self.sidecar_path}. "
                f"Run `add` first to create one."
            )
        self._sidecar = data
        self._manifests = {}
        for folder, info in data.get("blocks", {}).items():
            m = self.registry.find(folder)
            if m is None:
                raise UnknownBlockError(
                    f"Block '{folder}' is listed in the sidecar but was not "
                    f"found in {self.registry.blocks_dir}."
                )
            self._manifests[folder] = m
        return self

    def maybe_load(self) -> "ProjectState":
        """Like load() but returns silently if no sidecar exists."""
        if self.sidecar_path.exists():
            return self.load()
        self._sidecar = None
        self._manifests = {}
        return self

    def ensure_sidecar(self, blocks_dir: Path) -> dict:
        """Load existing sidecar or initialise a fresh one pointing at blocks_dir."""
        if self._sidecar is None:
            if self.sidecar_path.exists():
                self.load()
            else:
                self._sidecar = init_sidecar(blocks_dir)
                self._manifests = {}
        return self._sidecar

    def save_sidecar(self) -> Path:
        if self._sidecar is None:
            raise RuntimeError("No sidecar to save")
        return save_sidecar(self.project_dir, self._sidecar)

    def blocks(self) -> dict[str, BlockManifest]:
        return dict(self._manifests)

    def block_info(self, folder: str) -> dict | None:
        if self._sidecar is None:
            return None
        return self._sidecar.get("blocks", {}).get(folder)

    def add_block(
        self,
        folder: str,
        mode: str,
        explicit: bool,
        required_by: list[str] | None = None,
    ) -> bool:
        """Insert or update a block entry. Returns True if state changed."""
        assert self._sidecar is not None
        blocks = self._sidecar.setdefault("blocks", {})
        m = self.registry.find(folder)
        if m is None:
            raise UnknownBlockError(f"Block not found: {folder}")
        existing = blocks.get(folder)
        new_info = {
            "block_id": m.block_id,
            "mode": mode,
            "explicit": explicit,
            "required_by": sorted(set(required_by or [])),
        }
        if existing == new_info:
            return False
        blocks[folder] = new_info
        self._manifests[folder] = m
        return True

    def remove_block(self, folder: str) -> bool:
        """Drop a block entry. Returns True if state changed."""
        assert self._sidecar is not None
        blocks = self._sidecar.get("blocks", {})
        if folder not in blocks:
            return False
        del blocks[folder]
        self._manifests.pop(folder, None)
        # Strip the folder from any remaining required_by lists.
        for info in blocks.values():
            rb = info.get("required_by") or []
            if folder in rb:
                rb = [x for x in rb if x != folder]
                info["required_by"] = rb
        return True

    def installs(self) -> list[BlockInstall]:
        """Ordered list of (manifest, mode) pairs for every installed block."""
        out: list[BlockInstall] = []
        if self._sidecar is None:
            return out
        for folder in sorted(self._sidecar.get("blocks", {})):
            m = self._manifests.get(folder)
            if m is None:
                continue
            mode = self._sidecar["blocks"][folder].get("mode", MODE_REFERENCE)
            out.append(BlockInstall(manifest=m, mode=mode))
        return out

    def manifests(self) -> list[BlockManifest]:
        """Ordered list of installed manifests (sorted by folder name)."""
        return [self._manifests[k] for k in sorted(self._manifests)]

    # --- Apply sidecar -> project files ---

    def apply(self) -> bool:
        """Scaffold the project if needed, then refresh all region contents.

        Returns True if any file was written.
        """
        if not self.vcxproj_path.exists():
            raise FileNotFoundError(f"Missing vcxproj: {self.vcxproj_path}")
        editor = ProjectEditor(self.vcxproj_path).load()
        changed = False
        if not editor.is_scaffolded():
            editor.scaffold()
            changed = True

        installs = self.installs()
        macros = build_macros(installs)

        vcx = editor.vcxproj
        flt = editor.filters

        # Always replace region content, never remove the region. The scaffold
        # contract (see README) is that `remove` leaves the markers in place
        # holding an empty body so a subsequent `add` can repopulate them
        # without re-running scaffold(). The build_*_clcompile/clinclude
        # helpers already return an empty `<ItemGroup>` when given no installs,
        # so we just pass that through.
        new_vcx = vcx.text
        new_vcx = _replace_region_property_macros(new_vcx, macros)
        new_vcx = replace_region_content(
            new_vcx, REGION_CLCOMPILE, build_vcxproj_clcompile(installs)
        )
        new_vcx = replace_region_content(
            new_vcx, REGION_CLINCLUDE, build_vcxproj_clinclude(installs)
        )
        # Rewrite the vcxproj globals-properties region (empty content is OK;
        # markers stay holding an empty body until a <globalsProperty> addon
        # is installed).
        globals_existing = get_region_content(new_vcx, REGION_VCXPROJ_GLOBALS)
        if globals_existing is not None and globals_existing != macros.globals_lines:
            new_vcx = replace_region_content(
                new_vcx, REGION_VCXPROJ_GLOBALS, macros.globals_lines
            )
        if new_vcx != vcx.text:
            vcx.text = new_vcx
            changed = True

        new_flt = flt.text
        new_flt = replace_region_content(
            new_flt, REGION_FILTERS_FILTER, build_filters_filter(installs)
        )
        new_flt = replace_region_content(
            new_flt, REGION_FILTERS_CLCOMPILE, build_filters_clcompile(installs)
        )
        new_flt = replace_region_content(
            new_flt, REGION_FILTERS_CLINCLUDE, build_filters_clinclude(installs)
        )
        if new_flt != flt.text:
            flt.text = new_flt
            changed = True

        # Rewrite the addon-properties region in .props (empty content is OK;
        # the markers stay in place holding an empty body until an addon
        # contributes properties).
        props = editor.props
        if props.path.exists():
            new_props = props.text
            region_text = get_region_content(new_props, REGION_PROPS_PROPERTIES)
            if region_text is not None:
                # Normalise: content is `;`-joined `;`-separated lines.
                target = macros.props_lines
                if region_text != target:
                    new_props = replace_region_content(
                        new_props, REGION_PROPS_PROPERTIES, target
                    )
                    if new_props != props.text:
                        props.text = new_props
                        changed = True

        if changed:
            editor.save()
        return changed


def _replace_region_property_macros(text: str, macros: MacroValues) -> str:
    """Update the macro values inside the scaffold's PropertyGroup region."""
    content = get_region_content(text, REGION_PROPERTIES)
    if content is None:
        return text  # not scaffolded; nothing to update
    # Sub-element bodies are bounded by their open/close tags. We rewrite
    # each known tag's inner text. Use a small per-tag regex.
    replacements = {
        "CinderBlocksIncludePaths": macros.include_paths,
        "CinderBlocksLibPaths": macros.lib_paths,
        "CinderBlocksDebugLibs": macros.debug_libs,
        "CinderBlocksReleaseLibs": macros.release_libs,
        "CinderBlocksPostBuild": macros.post_build,
        "CinderBlocksDefines": macros.defines,
        "CinderBlocksAdditionalOptions": macros.additional_options,
        # CinderBlocksPostBuildChained is conditionally-defined; leave as-is.
    }
    new_content = content
    for tag, value in replacements.items():
        # Match <tag ...>inner</tag>; tolerate attribute on the open tag.
        pat = re.compile(
            rf"(<{tag}(?:\s[^>]*)?>)(.*?)(</{tag}>)",
            re.DOTALL,
        )
        def _repl(m: re.Match[str], value=value) -> str:
            return m.group(1) + value + m.group(3)

        new_content = pat.sub(_repl, new_content)
    return replace_region_content(text, REGION_PROPERTIES, new_content)


# ---------------------------------------------------------------------------
# Filesystem operations for copy mode
# ---------------------------------------------------------------------------
#
# Copy mode duplicates a block's directory tree into <project>/blocks/<folder>/.
# The cinderblock.xml <copyExclude> directive lists subdirectories to omit
# (typically `samples`, `test`, `proj`, `ci`); we honour it via a shutil
# ignore pattern. Re-copies (when the destination already exists) are handled
# by removing the destination first so the result mirrors the source exactly
# — important for `update` to pick up file removals upstream.


def _local_blocks_root(project_dir: Path) -> Path:
    return project_dir / "blocks"


def _local_block_dir(project_dir: Path, folder: str) -> Path:
    return _local_blocks_root(project_dir) / folder


def _copyignore_factory(copy_excludes: list[str]):
    """Build a shutil.ignore_patterns callable from <copyExclude> entries.

    Always also excludes `.git` (block sources are commonly their own clones
    but the version-control metadata is never useful inside the host project).
    """
    import fnmatch

    # User-provided excludes + always-on defaults. VCS metadata and other
    # common noise directories should never land in the host project tree.
    always_exclude = {".git", ".gitignore", ".github", ".gitmodules", ".svn", ".hg"}
    user_patterns = [p.strip() for p in copy_excludes if p.strip()]

    def _ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for n in names:
            if n in always_exclude:
                ignored.add(n)
                continue
            for pat in user_patterns:
                if fnmatch.fnmatch(n, pat) or n == pat:
                    ignored.add(n)
                    break
        return ignored

    return _ignore


def _robust_rmtree(path: Path) -> None:
    """shutil.rmtree that handles Windows' read-only files (e.g. .git/objects/pack/*.idx).

    Python 3.12+ supports the `onexc` callback; older versions used `onerror`.
    We clear the read-only bit before retrying the unlink.
    """
    import os
    import shutil
    import stat

    def _onexc(func, pathname, exc_info):
        # Best-effort recovery: chmod and retry once.
        try:
            os.chmod(pathname, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
        try:
            func(pathname)
        except OSError:
            # Last resort: defer until next reboot is not viable; just raise.
            raise

    try:
        shutil.rmtree(path, onexc=_onexc)
    except TypeError:
        # Python <3.12 fallback — uses onerror instead.
        def _onerror(func, pathname, exc_info):
            try:
                os.chmod(pathname, stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
            try:
                func(pathname)
            except OSError:
                raise

        shutil.rmtree(path, onerror=_onerror)


def copy_block_to_project(
    manifest: BlockManifest,
    project_dir: Path,
) -> Path:
    """Copy a block's source tree into <project>/blocks/<folder>/.

    Overwrites an existing copy so this is safe to call from `update`.
    Returns the destination directory.
    """
    src = manifest.block_dir
    if not src.is_dir():
        raise FileNotFoundError(f"Block source dir missing: {src}")
    dst = _local_block_dir(project_dir, manifest.folder_name)
    if dst.exists():
        _robust_rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(src, dst, ignore=_copyignore_factory(manifest.copy_excludes))
    return dst


def remove_block_from_project(project_dir: Path, folder: str) -> bool:
    """Delete <project>/blocks/<folder>/ if present. Returns True if removed."""
    dst = _local_block_dir(project_dir, folder)
    if not dst.exists():
        return False
    _robust_rmtree(dst)
    # If the parent blocks/ dir is now empty, tidy it up too.
    parent = dst.parent
    try:
        next(parent.iterdir())
    except StopIteration:
        parent.rmdir()
    return True


# ---------------------------------------------------------------------------
# add / remove / update command implementations
# ---------------------------------------------------------------------------


def cmd_add(args) -> int:
    blocks_dir = _resolve_blocks_dir(args.blocks_dir)
    try:
        registry = _build_registry(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    project_dir = Path(args.project).resolve()
    state = ProjectState(project_dir, registry).maybe_load()
    state.ensure_sidecar(blocks_dir)

    # Resolve dependencies transitively for every requested block.
    try:
        closure = resolve_deps(list(args.blocks), registry)
    except (UnknownBlockError, DependencyCycleError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    mode = args.mode
    changed_blocks: list[tuple[str, str]] = []
    copied_dirs: list[Path] = []

    if mode == MODE_COPY and not args.dry_run:
        # Pre-copy each unique block in the closure; deps inherit the same
        # mode as the requested block (TinderBox mirrors this behaviour).
        seen_for_copy: set[str] = set()
        for folder, _origin in closure:
            if folder in seen_for_copy:
                continue
            seen_for_copy.add(folder)
            m = registry.find(folder)
            if m is None:
                continue
            try:
                dst = copy_block_to_project(m, project_dir)
                copied_dirs.append(dst)
            except OSError as e:
                print(f"Error copying {folder}: {e}", file=sys.stderr)
                return 1

    for folder, origin in closure:
        if origin == "requested":
            explicit = True
            required_by: list[str] = []
        else:
            # origin is "dependency:<requiring_folder>"
            explicit = False
            required_by = [origin.split(":", 1)[1]]
        # Merge required_by into any pre-existing entry (e.g. two requesters
        # pull in the same dep).
        existing = state.block_info(folder)
        if existing is not None:
            required_by = sorted(
                set((existing.get("required_by") or [])) | set(required_by)
            )
            # If the block is already explicit, keep it explicit.
            explicit = explicit or bool(existing.get("explicit"))
        if state.add_block(folder, mode=mode, explicit=explicit, required_by=required_by):
            changed_blocks.append((folder, origin))

    if not changed_blocks:
        print("Nothing to do — requested blocks already installed.")
        return 0

    if args.dry_run:
        print("(dry run) would update sidecar and project files:")
        for folder, origin in changed_blocks:
            print(f"  + {folder:<24} ({origin}, {mode})")
        return 0

    state.save_sidecar()
    state.apply()

    # Report includes auto-installed deps so the user understands what
    # was pulled in.
    print(f"Updated {state.sidecar_path}")
    for folder, origin in changed_blocks:
        marker = "" if origin == "requested" else f" (auto: {origin})"
        print(f"  + {folder:<24}{marker}  [{mode}]")
    if copied_dirs:
        print(f"  copied {len(copied_dirs)} block(s) into {project_dir / 'blocks'}")
    return 0


def cmd_remove(args) -> int:
    blocks_dir = _resolve_blocks_dir(args.blocks_dir)
    try:
        registry = _build_registry(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    project_dir = Path(args.project).resolve()
    state = ProjectState(project_dir, registry).maybe_load()
    if state._sidecar is None:
        print(f"No sidecar JSON at {state.sidecar_path}; nothing to remove.")
        return 1
    state.load()

    # Check each requested block for reverse-dependencies.
    blocked: list[tuple[str, list[str]]] = []
    cascaded: list[str] = []
    for name in args.blocks:
        m = registry.find(name)
        if m is None:
            print(f"Error: block not found in registry: {name}", file=sys.stderr)
            return 1
        folder = m.folder_name
        info = state.block_info(folder)
        if info is None:
            print(f"(skip) {folder} is not installed.")
            continue
        required_by = info.get("required_by") or []
        if required_by and not args.force:
            blocked.append((folder, required_by))
            continue
        cascaded.append(folder)
        # Cascade: find blocks that became orphaned after removing `folder`.
        # An orphan is an implicit (explicit=false) block whose required_by
        # becomes empty once `folder` is gone.
        if args.keep_deps:
            continue
        for other_folder, other_info in list(state._sidecar["blocks"].items()):
            if other_folder == folder:
                continue
            if other_info.get("explicit"):
                continue
            other_rb = other_info.get("required_by") or []
            # Orphan check after this removal: would required_by only contain
            # things we're removing?
            survivors = [x for x in other_rb if x not in cascaded and x != folder]
            if not survivors and other_folder not in cascaded:
                cascaded.append(other_folder)

    if blocked:
        for folder, deps in blocked:
            print(
                f"Error: cannot remove {folder}, required by {', '.join(deps)}. "
                f"Use --force to override.",
                file=sys.stderr,
            )
        return 1

    if not cascaded:
        print("Nothing to remove.")
        return 0

    if args.dry_run:
        print("(dry run) would remove:")
        for folder in cascaded:
            info = state.block_info(folder) or {}
            print(f"  - {folder:<24} [{info.get('mode', MODE_REFERENCE)}]")
        return 0

    # Capture the mode of each block before removing it from the sidecar so
    # we know whether to clean up the copied directory.
    folder_modes = {}
    for folder in cascaded:
        info = state.block_info(folder) or {}
        folder_modes[folder] = info.get("mode", MODE_REFERENCE)
        state.remove_block(folder)
    state.save_sidecar()
    state.apply()

    # Remove copied block directories for any block that was in copy mode.
    removed_copies: list[str] = []
    for folder, m in folder_modes.items():
        if m == MODE_COPY:
            if remove_block_from_project(project_dir, folder):
                removed_copies.append(folder)

    print(f"Updated {state.sidecar_path}")
    for folder in cascaded:
        print(f"  - {folder}")
    if removed_copies:
        print(
            f"  deleted {len(removed_copies)} copied block dir(s): "
            + ", ".join(removed_copies)
        )
    return 0


def cmd_update(args) -> int:
    blocks_dir = _resolve_blocks_dir(args.blocks_dir)
    try:
        registry = _build_registry(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    project_dir = Path(args.project).resolve()
    state = ProjectState(project_dir, registry).maybe_load()
    if state._sidecar is None:
        print(f"No sidecar JSON at {state.sidecar_path}; nothing to update.")
        return 1
    state.load()

    # For copy-mode blocks, re-sync the local copy so new/removed files
    # upstream are reflected. Reference-mode blocks are read straight from
    # the source tree, so they're always up to date.
    synced_copies: list[str] = []
    if not args.dry_run:
        for inst in state.installs():
            if inst.mode == MODE_COPY:
                copy_block_to_project(inst.manifest, project_dir)
                synced_copies.append(inst.manifest.folder_name)

    if args.dry_run:
        print("(dry run) would re-derive region contents from installed blocks.")
        return 0

    changed = state.apply()
    if changed:
        print(f"Updated project files in {project_dir / 'vc2022'}")
    elif synced_copies:
        print(f"Re-synced {len(synced_copies)} copy-mode block(s) "
              f"({', '.join(synced_copies)}).")
    else:
        print("Already in sync.")
    return 0


def cmd_list(args) -> int:
    blocks_dir = _resolve_blocks_dir(args.blocks_dir)

    project_dir = Path(args.project).resolve() if args.project else None
    installed: dict[str, dict] = {}
    if project_dir:
        sidecar = load_sidecar(project_dir)
        if sidecar is not None:
            installed = sidecar.get("blocks", {})
        elif args.project and not args.available:
            print(
                f"(no {SIDECAR_FILENAME} in {project_dir / 'vc2022'})",
                file=sys.stderr,
            )

    if args.available:
        try:
            registry = _build_registry(args)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        msw_blocks = [m for m in registry.blocks.values() if m.supports_msw]
        other_blocks = [m for m in registry.blocks.values() if not m.supports_msw]

        print(f"# Available blocks in {blocks_dir}")
        print(
            f"# {len(msw_blocks)} MSW-supported, "
            f"{len(other_blocks)} non-MSW, "
            f"{len(registry.errors)} parse errors"
        )
        print()
        for folder, m in sorted((b.folder_name, b) for b in msw_blocks):
            mark = "+" if folder in installed else " "
            req = f"  requires: {', '.join(m.requires)}" if m.requires else ""
            print(f"  [{mark}] {folder:<24} {m.display_name}  -  {m.summary}{req}")

        if other_blocks:
            print()
            print(f"# Excluded (non-MSW): {len(other_blocks)}")
            for m in sorted(other_blocks, key=lambda x: x.folder_name):
                os_list = ",".join(sorted(m.supported_os)) or "(unspecified)"
                print(f"      {m.folder_name:<24} os={os_list}")

        if registry.errors:
            print()
            print(f"# Parse errors: {len(registry.errors)}")
            for folder, err in sorted(registry.errors.items()):
                print(f"      {folder}: {err}")
        return 0

    # Default: list installed
    if not installed:
        print("No blocks installed in this project.")
        print("Run with --available to see installable blocks.")
        return 0

    print(f"# Installed blocks in {project_dir}")
    print(f"# {len(installed)} total")
    print()
    for name, info in installed.items():
        mode = info.get("mode", "?")
        explicit = info.get("explicit", True)
        required_by = info.get("required_by", [])
        if explicit:
            origin = "explicit"
        elif required_by:
            origin = f"auto (required by {', '.join(required_by)})"
        else:
            origin = "auto"
        print(f"  {name:<24} [{mode}]  {origin}")
    return 0


def cmd_resolve(args) -> int:
    blocks_dir = _resolve_blocks_dir(args.blocks_dir)
    try:
        registry = _build_registry(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        closure = resolve_deps(args.blocks, registry)
    except (UnknownBlockError, DependencyCycleError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("# Resolved dependency closure (topological order):")
    for folder, origin in closure:
        print(f"  {folder:<24}  ({origin})")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cinder_blocks",
        description="Manage Cinder blocks in NewTypeEngine projects.",
    )
    parser.add_argument(
        "--blocks-dir",
        default=None,
        help="Path to Cinder blocks directory "
        "(default: env CINDER_BLOCKS_DIR or built-in default).",
    )
    parser.add_argument(
        "--engine-addons-dir",
        default=None,
        help="Path to engine addons directory "
        "(default: env NT_ENGINE_ADDONS_DIR or <engine>/engine_addons/).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List installed or available blocks")
    p_list.add_argument(
        "--project",
        default=None,
        help="Path to NewTypeEngine project root (for installed-block view).",
    )
    p_list.add_argument(
        "--available",
        action="store_true",
        help="List all blocks in blocks-dir, marking installed ones.",
    )
    p_list.set_defaults(func=cmd_list)

    p_resolve = sub.add_parser(
        "resolve",
        help="Resolve transitive dependencies of named blocks (debug helper).",
    )
    p_resolve.add_argument("blocks", nargs="+", help="Block folder names, ids, or display names.")
    p_resolve.set_defaults(func=cmd_resolve)

    p_add = sub.add_parser(
        "add",
        help="Install one or more blocks into a NewTypeEngine project.",
    )
    p_add.add_argument("blocks", nargs="+", help="Block folder names, ids, or display names.")
    p_add.add_argument(
        "--project",
        required=True,
        help="Path to NewTypeEngine project root (must contain vc2022/).",
    )
    p_add.add_argument(
        "--mode",
        choices=("reference", "copy"),
        default="reference",
        help="reference = emit paths via $(CinderBlocksDir); "
        "copy = duplicate block files into <project>/blocks/ (P5).",
    )
    p_add.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing anything.",
    )
    p_add.set_defaults(func=cmd_add)

    p_remove = sub.add_parser(
        "remove",
        help="Remove one or more blocks from a project.",
    )
    p_remove.add_argument("blocks", nargs="+", help="Block folder names.")
    p_remove.add_argument(
        "--project", required=True,
        help="Path to NewTypeEngine project root.",
    )
    p_remove.add_argument(
        "--force", action="store_true",
        help="Remove even if other installed blocks depend on this one.",
    )
    p_remove.add_argument(
        "--keep-deps", action="store_true",
        help="Do not auto-remove dependencies that would become orphaned.",
    )
    p_remove.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing anything.",
    )
    p_remove.set_defaults(func=cmd_remove)

    p_update = sub.add_parser(
        "update",
        help="Re-derive vcxproj region contents from the installed block set.",
    )
    p_update.add_argument(
        "--project", required=True,
        help="Path to NewTypeEngine project root.",
    )
    p_update.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing anything.",
    )
    p_update.set_defaults(func=cmd_update)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
