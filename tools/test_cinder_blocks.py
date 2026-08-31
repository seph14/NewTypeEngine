#!/usr/bin/env python3
"""Tests for tools/cinder_blocks.py.

P1 tests are integration tests against the real Cinder blocks at
C:\\Users\\barca\\Projects\\CinderHead\\blocks — they pin the parser's
behaviour to actual block metadata so future refactors catch regressions.
The pinned path here is independent of `cb.DEFAULT_CINDER_BLOCKS_DIR` (which
points at the runtime Cinder\\blocks tree to match the .props <CinderRoot>):
the parser tests need a richer block set to exercise every XML variant, and
CinderHead is the tree that has it. The CLI tests below use the runtime
default, exercising the production path.

P2 tests cover the vcxproj/filters/props editors. The P2 exit criterion is
byte-identity: scaffold followed by unscaffold on a freshly-generated
project must yield bytes identical to the original.

Run with:
    python -m unittest tools.test_cinder_blocks -v
or:
    python tools/test_cinder_blocks.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Make the module importable when run as a script or via -m unittest.
TOOLS_DIR = Path(__file__).resolve().parent
ENGINE_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(ENGINE_ROOT))

from tools import cinder_blocks as cb  # noqa: E402

# Pinned to CinderHead (not the runtime default Cinder\blocks) because the
# parser tests need blocks like Cinder-WebSocketPP, Cinder-NDI, Cinder-URG,
# AX-MediaPlayer etc. to exercise every XML variant. The CLI tests below
# use cb._resolve_blocks_dir(None) which falls through to the runtime
# default. See the module docstring.
BLOCKS_DIR = Path(r"C:\Users\barca\Projects\CinderHead\blocks")
ENGINE_ADDONS_DIR = ENGINE_ROOT / "engine_addons"


def _full_registry() -> cb.BlockRegistry:
    """Registry spanning both the Cinder blocks dir and the engine addons dir."""
    return cb.BlockRegistry(BLOCKS_DIR, extra_dirs=[ENGINE_ADDONS_DIR]).scan()


def _registry() -> cb.BlockRegistry:
    return cb.BlockRegistry(BLOCKS_DIR).scan()


# ---------------------------------------------------------------------------
# Parser — per-block pinning against real cinderblock.xml files
# ---------------------------------------------------------------------------


class TestParserEdgeCases(unittest.TestCase):
    """Verify the parser handles every XML variant observed in the survey."""

    @classmethod
    def setUpClass(cls):
        cls.reg = _registry()

    def test_registry_scanned_without_fatal_errors(self):
        # The scan collects per-block errors but should not crash.
        self.assertGreater(len(self.reg.blocks), 20)
        # Cinder-URG vs Cinder-LDLidar both claim org.libcinder.urg — known
        # metadata bug. Confirmed as a parse-error entry, not a fatal one.
        self.assertIn("Cinder-URG", self.reg.errors)

    def test_block_with_single_source_and_header(self):
        # Cairo uses <source> (singular) and <header> (singular) at top level.
        m = self.reg.find("Cairo")
        self.assertIsNotNone(m)
        self.assertIn("src/Cairo.cpp", [s.path for s in m.sources])
        self.assertIn("include/cinder/cairo/Cairo.h", [h.path for h in m.headers])

    def test_block_with_multiple_single_sources(self):
        # Cinder-Warping lists four <source> entries and one <header>.
        m = self.reg.find("Cinder-Warping")
        srcs = [s.path for s in m.sources]
        self.assertIn("src/Warp.cpp", srcs)
        self.assertIn("src/WarpPerspectiveBilinear.cpp", srcs)
        self.assertEqual(len(srcs), 4)
        self.assertIn("include/Warp.h", [h.path for h in m.headers])

    def test_block_with_source_compile_as_attr(self):
        # LocationManager uses <source compileAs="mm">. It's iOS/macOS-only,
        # so the block should be excluded by supports_msw, but parsing should
        # still capture the attribute if we parse directly.
        m = cb.parse_manifest(BLOCKS_DIR / "LocationManager")
        src = next(s for s in m.sources if s.path == "src/cinder/LocationManager.cpp")
        self.assertEqual(src.compile_as, "mm")

    def test_glob_pattern_expansion(self):
        # OSC: src/cinder/osc/*.cpp expands to exactly Osc.cpp.
        m = self.reg.find("OSC")
        self.assertEqual([s.path for s in m.sources], ["src/cinder/osc/Osc.cpp"])
        self.assertEqual([h.path for h in m.headers], ["src/cinder/osc/Osc.h"])

    def test_glob_recursive_subdir_pattern(self):
        # Box2D uses patterns like src/Box2D/Dynamics/Contacts/*.cpp.
        m = self.reg.find("Box2D")
        srcs = {s.path for s in m.sources}
        self.assertIn("src/Box2D/Dynamics/Contacts/b2ContactSolver.cpp", srcs)
        # Multiple subdirectories under src/Box2D/* should be captured per
        # pattern, but each pattern only matches its own subdirectory.
        self.assertTrue(any(p.startswith("src/Box2D/Dynamics/Joints/") for p in srcs))
        self.assertTrue(any(p.startswith("src/Box2D/Common/") for p in srcs))

    def test_include_path_with_cinder_attr(self):
        # Cinder-WebSocketPP: <includePath cinder="true">include/asio</includePath>
        # marks a path relative to Cinder root, not the block.
        m = self.reg.find("Cinder-WebSocketPP")
        asio = next(ip for ip in m.include_paths if ip.path == "include/asio")
        self.assertTrue(asio.cinder)
        # Regular <includePath system="true">src</includePath> should keep
        # cinder=False.
        src_ip = next(ip for ip in m.include_paths if ip.path == "src")
        self.assertTrue(src_ip.system)
        self.assertFalse(src_ip.cinder)

    def test_static_library_absolute_attr(self):
        # OpenCV4 has <staticLibrary absolute="true">-lz</staticLibrary> on mac,
        # but that's a mac-only branch. On msw, OpenCV4 has config-specific
        # libs instead. Cairo on msw has <staticLibrary absolute="true">libpng.lib.
        m = self.reg.find("Cairo")
        absolute_libs = [lib for lib in m.static_libs if lib.absolute]
        self.assertIn(
            "libpng.lib",
            [lib.path for lib in absolute_libs],
        )

    def test_config_specific_static_libraries(self):
        # Cinder-URG places debug/release variants inside nested <platform>.
        m = self.reg.find("Cinder-URG")
        debug_libs = [lib for lib in m.static_libs if lib.config == "debug"]
        release_libs = [lib for lib in m.static_libs if lib.config == "release"]
        universal_libs = [lib for lib in m.static_libs if lib.config is None]
        self.assertEqual(
            [lib.path for lib in debug_libs], ["sdk/$(PlatformTarget)/urg_cpp_debug.lib"]
        )
        self.assertEqual(
            [lib.path for lib in release_libs], ["sdk/$(PlatformTarget)/urg_cpp.lib"]
        )
        # setupapi.lib sits at the top level inside <platform os="msw">,
        # outside any config branch — should be tagged config=None.
        self.assertIn("setupapi.lib", [lib.path for lib in universal_libs])

    def test_build_copy_collected(self):
        # Cinder-NDI ships Processing.NDI.Lib.x64.dll via <buildCopy>.
        m = self.reg.find("Cinder-NDI")
        self.assertTrue(
            any("Processing.NDI.Lib" in p and p.endswith(".dll") for p in m.build_copies)
        )

    def test_preprocessor_define_collected(self):
        # Cinder-LibArtnet declares HAVE_CONFIG_H. It's non-MSW so excluded
        # from the registry's MSW view — parse the file directly.
        m = cb.parse_manifest(BLOCKS_DIR / "Cinder-LibArtnet")
        self.assertIn("HAVE_CONFIG_H", m.preprocessor_defines)

    def test_copy_exclude_collected(self):
        # Cinder-Serial uses <copyExclude>samples</copyExclude>.
        m = self.reg.find("Cinder-Serial")
        self.assertIn("samples", m.copy_excludes)

    def test_msw_platform_branch_picked_up(self):
        # AX-MediaPlayer combines top-level patterns + <platform os="msw">.
        # Top-level: src/*.cxx, src/*.h, src (include path)
        # msw branch: src/msw/*.cxx, src/msw/*.h, src/msw (include path)
        m = self.reg.find("AX-MediaPlayer")
        inc_paths = [ip.path for ip in m.include_paths]
        self.assertIn("src", inc_paths)
        self.assertIn("src/msw", inc_paths)
        # macosx branch must NOT leak in.
        self.assertNotIn("src/osx", inc_paths)

    def test_non_msw_platform_branch_excluded(self):
        # Cairo has <platform os="macosx"> with libcairo.a etc. On MSW we
        # should pick up lib/msw/x86/cairo-static.lib and not the mac libs.
        m = self.reg.find("Cairo")
        lib_paths = [lib.path for lib in m.static_libs]
        self.assertIn("lib/msw/x86/cairo-static.lib", lib_paths)
        self.assertFalse(any("macosx" in p for p in lib_paths))

    def test_xml_comments_ignored(self):
        # Cinder-FFmpeg has commented-out <staticLibrary> and <buildCopy>
        # entries inside its <platform os="msw">. ElementTree naturally
        # skips comments, but assert the active libs/dlls are present and
        # the commented ones (avdevice, avfilter) are not.
        m = self.reg.find("Cinder-FFmpeg")
        lib_paths = [lib.path for lib in m.static_libs]
        self.assertIn("build/lib/msw/$(PlatformTarget)/avcodec.lib", lib_paths)
        self.assertNotIn("build/lib/msw/$(PlatformTarget)/avdevice.lib", lib_paths)
        self.assertNotIn("build/lib/msw/$(PlatformTarget)/avfilter.lib", lib_paths)
        copies = list(m.build_copies)
        self.assertTrue(any("avcodec-58.dll" in c for c in copies))
        self.assertFalse(any("avdevice-58.dll" in c for c in copies))

    def test_typo_in_include_path_preserved_verbatim(self):
        # Clipper has <includePath>incude</includePath> (sic). The tool
        # should not "fix" metadata — emit verbatim and let MSBuild fail.
        m = self.reg.find("Clipper")
        self.assertIn("incude", [ip.path for ip in m.include_paths])

    def test_framework_ignored_on_msw(self):
        # OpenCV3 has <framework sdk="true">OpenCL.framework</framework> in
        # the mac branch, which never reaches MSW. Top-level OpenCV3 also
        # has the framework in mac — confirm nothing leaks through as a
        # static lib or include path that looks like a framework.
        m = self.reg.find("Cinder-OpenCV3")
        # Sanity: msw branch has individual debug libs (opencv_core300d.lib etc.)
        # since OpenCV3 splits into separate modules rather than opencv_world.
        lib_paths = [lib.path for lib in m.static_libs]
        self.assertTrue(any("opencv_core300d.lib" in p for p in lib_paths))
        self.assertTrue(any("opencv_calib3d300d.lib" in p for p in lib_paths))

    def test_supports_msw_property(self):
        # LocationManager is iOS/macOS-only and should report supports_msw=False.
        m = cb.parse_manifest(BLOCKS_DIR / "LocationManager")
        self.assertFalse(m.supports_msw)
        # OSC has no <supports> tag and is universal.
        osc = self.reg.find("OSC")
        self.assertTrue(osc.supports_msw)

    def test_requires_field_captured(self):
        m = self.reg.find("TUIO")
        self.assertEqual(m.requires, ["org.libcinder.osc"])


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


class TestRegistryLookup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = _registry()

    def test_find_by_folder_name(self):
        self.assertIsNotNone(self.reg.find("OSC"))
        self.assertIsNotNone(self.reg.find("Cinder-NDI"))

    def test_find_by_block_id(self):
        self.assertIsNotNone(self.reg.find("org.libcinder.osc"))
        self.assertIsNotNone(self.reg.find("org.libcinder.ndi"))

    def test_find_case_insensitive(self):
        # OSC's id is `org.libcinder.OSC` but TUIO requires `org.libcinder.osc`.
        # Both spellings must resolve to the same block.
        a = self.reg.find("org.libcinder.OSC")
        b = self.reg.find("org.libcinder.osc")
        self.assertIsNotNone(a)
        self.assertIs(a, b)
        # Folder-name lookup should also be case-insensitive.
        self.assertIs(self.reg.find("osc"), self.reg.find("OSC"))

    def test_find_by_display_name(self):
        # The XML name attribute for Cinder-WMFVideo is "WMFVideo".
        m = self.reg.find("WMFVideo")
        self.assertIsNotNone(m)
        self.assertEqual(m.folder_name, "Cinder-WMFVideo")

    def test_find_unknown_returns_none(self):
        self.assertIsNone(self.reg.find("DoesNotExist"))
        self.assertIsNone(self.reg.find("org.fake.id"))


# ---------------------------------------------------------------------------
# Dependency resolver
# ---------------------------------------------------------------------------


class TestResolveDeps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = _registry()

    def test_tuio_pulls_in_osc(self):
        closure = cb.resolve_deps(["TUIO"], self.reg)
        folders = [name for name, _ in closure]
        self.assertEqual(folders, ["OSC", "TUIO"])
        self.assertEqual(closure[0][1], "dependency:TUIO")
        self.assertEqual(closure[1][1], "requested")

    def test_shared_dependency_deduplicated(self):
        # SjPathTracer and SjPBRUtil both require SjRenderLib; closure should
        # contain it exactly once, before both requesters.
        closure = cb.resolve_deps(["SjPathTracer", "SjPBRUtil"], self.reg)
        folders = [name for name, _ in closure]
        self.assertEqual(folders.count("SjRenderLib"), 1)
        self.assertLess(folders.index("SjRenderLib"), folders.index("SjPathTracer"))
        self.assertLess(folders.index("SjRenderLib"), folders.index("SjPBRUtil"))

    def test_block_without_deps_returns_self(self):
        closure = cb.resolve_deps(["Cinder-NDI"], self.reg)
        self.assertEqual([name for name, _ in closure], ["Cinder-NDI"])

    def test_unknown_block_raises(self):
        with self.assertRaises(cb.UnknownBlockError):
            cb.resolve_deps(["Nope"], self.reg)

    def test_unknown_required_id_raises(self):
        # Construct a synthetic registry with one block that requires a
        # non-existent id. cycle-style scenario without modifying real data.
        class _StubRegistry:
            def find(self, name_or_id):
                if name_or_id.lower() == "a":
                    m = cb.BlockManifest(
                        folder_name="A",
                        display_name="A",
                        block_dir=BLOCKS_DIR,
                        requires=["org.does.not.exist"],
                    )
                    return m
                return None

        with self.assertRaises(cb.UnknownBlockError) as ctx:
            cb.resolve_deps(["A"], _StubRegistry())
        self.assertIn("org.does.not.exist", str(ctx.exception))

    def test_cycle_detected(self):
        # Build a synthetic cycle: A -> B -> A.
        class _CycleRegistry:
            def __init__(self):
                self._a = cb.BlockManifest(
                    folder_name="A", display_name="A", block_dir=BLOCKS_DIR,
                    requires=["b.id"],
                )
                self._b = cb.BlockManifest(
                    folder_name="B", display_name="B", block_dir=BLOCKS_DIR,
                    requires=["a.id"],
                )

            def find(self, name_or_id):
                key = name_or_id.lower()
                if key in ("a", "a.id"):
                    return self._a
                if key in ("b", "b.id"):
                    return self._b
                return None

        with self.assertRaises(cb.DependencyCycleError):
            cb.resolve_deps(["A"], _CycleRegistry())


# ---------------------------------------------------------------------------
# Sanity: load_sidecar with no project returns None cleanly
# ---------------------------------------------------------------------------


class TestSidecar(unittest.TestCase):
    def test_missing_sidecar_returns_none(self):
        # A temp dir with no vc2022/.cinder-blocks.json should yield None.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(cb.load_sidecar(Path(tmp)))


# ===========================================================================
# P2: Region primitives
# ===========================================================================


class TestRegionPrimitives(unittest.TestCase):
    """Cover the marker-region helpers in isolation."""

    def test_ensure_then_find_then_remove_round_trip(self):
        text = (
            '<Root>\n'
            '  <Header />\n'
            '  <Anchor />\n'
            '  <Footer />\n'
            '</Root>\n'
        )
        original = text
        text = cb.ensure_region(text, "test-region", "<Anchor />", content="hello")
        self.assertIsNotNone(cb.find_region(text, "test-region"))
        self.assertIn("hello", cb.get_region_content(text, "test-region"))
        text = cb.remove_region(text, "test-region")
        self.assertEqual(text, original)

    def test_ensure_region_idempotent(self):
        text = "<Root>\n<Anchor />\n</Root>\n"
        a = cb.ensure_region(text, "r", "<Anchor />", content="x")
        b = cb.ensure_region(a, "r", "<Anchor />", content="x")
        self.assertEqual(a, b)

    def test_replace_region_content(self):
        text = "<Root>\n<Anchor />\n</Root>\n"
        text = cb.ensure_region(text, "r", "<Anchor />", content="old")
        text = cb.replace_region_content(text, "r", "new")
        self.assertIn("new", cb.get_region_content(text, "r"))
        self.assertNotIn("old", cb.get_region_content(text, "r"))

    def test_remove_region_when_absent_is_noop(self):
        text = "<Root/>\n"
        self.assertEqual(cb.remove_region(text, "missing"), text)

    def test_region_preserves_crlf(self):
        # Round-trip must work when the host file uses CRLF.
        text = "<Root>\r\n<Anchor />\r\n</Root>\r\n"
        original = text
        out = cb.ensure_region(text, "r", "<Anchor />", content="x")
        self.assertNotEqual(out, original)
        out = cb.remove_region(out, "r")
        self.assertEqual(out, original)

    def test_ensure_region_missing_anchor_raises(self):
        with self.assertRaises(ValueError):
            cb.ensure_region("<Root/>\n", "r", "no-such-anchor")


class TestInjectSuffix(unittest.TestCase):
    def test_inject_into_matching_tag(self):
        text = "<Root>\n<Foo>a</Foo>\n<Foo>b</Foo>\n</Root>\n"
        out = cb.inject_suffix_before_tag(text, "Foo", "_X")
        self.assertEqual(out.count("_X</Foo>"), 2)

    def test_inject_idempotent(self):
        text = "<Root>\n<Foo>a</Foo>\n</Root>\n"
        once = cb.inject_suffix_before_tag(text, "Foo", "_X")
        twice = cb.inject_suffix_before_tag(once, "Foo", "_X")
        self.assertEqual(once, twice)

    def test_uninject_reverses(self):
        text = "<Root>\n<Foo>a</Foo>\n</Root>\n"
        out = cb.inject_suffix_before_tag(text, "Foo", "_X")
        out = cb.uninject_suffix_before_tag(out, "Foo", "_X")
        self.assertEqual(out, text)

    def test_inject_with_restrict_to(self):
        # Inject only into elements whose enclosing ItemDefinitionGroup
        # condition contains the restrict_to substring.
        text = (
            '<ItemDefinitionGroup Condition="A==\'Debug\'"><Foo>a</Foo></ItemDefinitionGroup>\n'
            '<ItemDefinitionGroup Condition="A==\'Release\'"><Foo>b</Foo></ItemDefinitionGroup>\n'
        )
        out = cb.inject_suffix_before_tag(
            text, "Foo", "_DEBUG", restrict_to="=='Debug'"
        )
        self.assertEqual(out.count("_DEBUG</Foo>"), 1)
        # The Release Foo should not be touched.
        self.assertIn("b</Foo>", out)


# ===========================================================================
# P2: TextFile newline preservation
# ===========================================================================


class TestTextFile(unittest.TestCase):
    def test_crlf_preserved_on_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_bytes(b"line1\r\nline2\r\n")
            tf = cb.TextFile(p)
            tf.load()
            # Edit while in \n-normalised form.
            tf.text = tf.text + "line3\n"
            tf.save()
            self.assertEqual(p.read_bytes(), b"line1\r\nline2\r\nline3\r\n")

    def test_lf_preserved_on_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_bytes(b"line1\nline2\n")
            tf = cb.TextFile(p)
            tf.load()
            tf.text = tf.text + "line3\n"
            tf.save()
            self.assertEqual(p.read_bytes(), b"line1\nline2\nline3\n")


# ===========================================================================
# P2: ProjectEditor round-trip on a freshly-generated project
# ===========================================================================


def _generate_project(tmp: Path, name: str) -> Path:
    """Invoke tools/generate_project.py to create a real project under tmp."""
    subprocess.run(
        [sys.executable, str(ENGINE_ROOT / "tools" / "generate_project.py"),
         "--path", str(tmp), "--name", name],
        check=True,
        cwd=ENGINE_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tmp / name / "vc2022" / f"{name}.vcxproj"


class _SameDriveTempDir:
    """Context manager yielding a temp dir on the same drive as ENGINE_ROOT.

    tools/generate_project.py uses os.path.relpath, which raises ValueError
    when start and target are on different Windows drives. Default tempfile
    placement on C: would break for an engine rooted on D:, so we put the
    temp dir under the engine's own build/ tree instead.
    """

    def __init__(self):
        self._tmp = None

    def __enter__(self) -> Path:
        base = ENGINE_ROOT / "build" / "_test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        import tempfile

        self._tmp = Path(tempfile.mkdtemp(dir=base))
        return self._tmp

    def __exit__(self, *exc):
        import shutil

        if self._tmp and self._tmp.exists():
            shutil.rmtree(self._tmp, ignore_errors=True)


class TestProjectEditorRoundTrip(unittest.TestCase):
    """The P2 exit criterion: scaffold -> unscaffold yields byte-identity."""

    def test_round_trip_preserves_vcxproj_bytes(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "RoundTrip")
            original = vcx.read_bytes()
            self.assertIn(b"</Project>", original)

            editor = cb.ProjectEditor(vcx).load()
            self.assertFalse(editor.is_scaffolded())
            self.assertTrue(editor.scaffold())
            self.assertTrue(editor.is_scaffolded())
            scaffolded = vcx.read_bytes()
            self.assertNotEqual(scaffolded, original)

            self.assertTrue(editor.unscaffold())
            self.assertFalse(editor.is_scaffolded())
            final = vcx.read_bytes()
            self.assertEqual(
                final,
                original,
                "vcxproj bytes differ after scaffold->unscaffold round-trip",
            )

    def test_round_trip_preserves_filters_bytes(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "RoundTripFilters")
            filters = vcx.with_suffix(".vcxproj.filters")
            original = filters.read_bytes()

            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            self.assertNotEqual(filters.read_bytes(), original)
            editor.unscaffold()
            self.assertEqual(filters.read_bytes(), original)

    def test_round_trip_preserves_props_bytes(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "RoundTripProps")
            props = vcx.with_suffix(".props")
            original = props.read_bytes()

            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            self.assertNotEqual(props.read_bytes(), original)
            self.assertIn(
                b"<CinderBlocksDir>",
                props.read_bytes(),
            )
            editor.unscaffold()
            self.assertEqual(props.read_bytes(), original)

    def test_scaffold_idempotent(self):
        # Scaffolding an already-scaffolded project is a no-op.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "Idempotent")
            editor = cb.ProjectEditor(vcx).load()
            self.assertTrue(editor.scaffold())
            after_first = vcx.read_bytes()
            self.assertFalse(editor.scaffold())  # no change
            self.assertEqual(vcx.read_bytes(), after_first)

    def test_unscaffold_when_not_scaffolded_is_noop(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "NoopUn")
            editor = cb.ProjectEditor(vcx).load()
            self.assertFalse(editor.unscaffold())  # nothing to undo

    def test_scaffold_includes_expected_macro_references(self):
        # Spot-check that scaffold injects the property references into the
        # right ItemDefinitionGroups.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "ScaffoldCheck")
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            text = vcx.read_text(encoding="utf-8")
            # All ItemDefinitionGroups should reference the include paths macro.
            self.assertEqual(
                text.count(";$(CinderBlocksIncludePaths)"),
                text.count("<AdditionalIncludeDirectories>"),
            )
            # Debug|x64 and Debug_Runtime|x64 both get the debug libs macro.
            self.assertGreaterEqual(text.count(";$(CinderBlocksDebugLibs)"), 2)
            # Release gets the release libs macro.
            self.assertGreaterEqual(text.count(";$(CinderBlocksReleaseLibs)"), 1)
            # PostBuild gets the chained macro.
            self.assertEqual(
                text.count(" $(CinderBlocksPostBuildChained)"),
                text.count("<Command>"),
            )


# ===========================================================================
# P6: Engine addons — parser, builders, scaffold, end-to-end cmd_add
# ===========================================================================
#
# Exercises the three schema extensions (`absolute="true"` on
# source/header/includePath, `<property>` element, `preprocessorDefine`
# emission) against the two real engine addon manifests at
# `engine_addons/{sim,video}/cinderblock.xml`.


class TestEngineAddonParser(unittest.TestCase):
    """Verify the parser handles absolute="true" + <property> + <preprocessorDefine>."""

    @classmethod
    def setUpClass(cls):
        cls.reg = _full_registry()

    def test_registry_picks_up_engine_addons(self):
        # The engine addons dir is purely additive — both sim and video
        # should resolve alongside Cinder blocks like OSC.
        self.assertIsNotNone(self.reg.find("sim"))
        self.assertIsNotNone(self.reg.find("video"))
        self.assertIsNotNone(self.reg.find("OSC"))

    def test_sim_sources_are_absolute(self):
        m = self.reg.find("sim")
        self.assertEqual(len(m.sources), 1)
        self.assertTrue(m.sources[0].absolute)
        self.assertEqual(
            m.sources[0].path,
            r"..\src\newtype\physics\Physics.cpp",
        )

    def test_sim_headers_are_absolute(self):
        m = self.reg.find("sim")
        self.assertEqual(len(m.headers), 1)
        self.assertTrue(m.headers[0].absolute)

    def test_sim_include_paths_are_absolute(self):
        m = self.reg.find("sim")
        for ip in m.include_paths:
            self.assertTrue(ip.absolute, f"{ip.path} should be absolute")
        self.assertEqual(
            [ip.path for ip in m.include_paths],
            ["$(LCSInclude)", "$(LCSIncludeLcpp)", "$(LCSIncludeEigen)", "$(LCSIncludeYyjson)"],
        )

    def test_sim_parses_preprocessor_define(self):
        m = self.reg.find("sim")
        self.assertEqual(m.preprocessor_defines, ["$(LCSNoInternalFiberScheduler)"])

    def test_sim_parses_properties(self):
        m = self.reg.find("sim")
        names = [p.name for p in m.properties]
        self.assertIn("LCSRoot", names)
        self.assertIn("LCSInclude", names)
        self.assertIn("LCSNoInternalFiberScheduler", names)
        # Property count: 8 (LCSRoot + 4 includes + 2 lib paths + scheduler define)
        self.assertEqual(len(m.properties), 8)
        # Verify a value that references another declared property.
        lcs_inc = next(p for p in m.properties if p.name == "LCSInclude")
        self.assertEqual(lcs_inc.value, r"$(LCSRoot)\Solver")

    def test_sim_static_libs_split_debug_release(self):
        m = self.reg.find("sim")
        debug = [l for l in m.static_libs if l.config == "debug"]
        release = [l for l in m.static_libs if l.config == "release"]
        self.assertEqual(len(debug), 1)
        self.assertEqual(len(release), 1)
        self.assertTrue(debug[0].absolute)
        self.assertTrue(release[0].absolute)
        self.assertIn(r"$(LCSLibDebug)", debug[0].path)
        self.assertIn(r"$(LCSLibRelease)", release[0].path)

    def test_video_sources_absolute(self):
        m = self.reg.find("video")
        self.assertEqual(len(m.sources), 3)
        for s in m.sources:
            self.assertTrue(s.absolute)

    def test_video_headers_absolute(self):
        m = self.reg.find("video")
        # 4 headers including NativeTextureDesc.h
        self.assertEqual(len(m.headers), 4)
        names = [h.path for h in m.headers]
        self.assertIn(r"..\include\newtype\media\NativeTextureDesc.h", names)
        for h in m.headers:
            self.assertTrue(h.absolute)

    def test_video_libs_universal_absolute(self):
        m = self.reg.find("video")
        # All 6 link libs are universal (no config tag) and absolute.
        paths = [l.path for l in m.static_libs]
        for lib in ("d3d11.lib", "dxguid.lib", "mf.lib", "mfplat.lib", "mfreadwrite.lib", "mfuuid.lib"):
            self.assertIn(lib, paths)
        for l in m.static_libs:
            self.assertTrue(l.absolute)
            self.assertIsNone(l.config)


class TestEngineAddonMacroBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = _full_registry()

    def _installs(self, *folders, mode=cb.MODE_REFERENCE):
        return [cb.BlockInstall(manifest=self.reg.find(f), mode=mode) for f in folders]

    def test_sim_macros_include_lcs_paths(self):
        m = self.reg.find("sim")
        macros = cb.build_macros([cb.BlockInstall(manifest=m)])
        self.assertIn("$(LCSInclude)", macros.include_paths)
        self.assertIn("$(LCSIncludeLcpp)", macros.include_paths)
        self.assertIn("$(LCSIncludeEigen)", macros.include_paths)
        self.assertIn("$(LCSIncludeYyjson)", macros.include_paths)

    def test_sim_macros_emit_defines(self):
        m = self.reg.find("sim")
        macros = cb.build_macros([cb.BlockInstall(manifest=m)])
        self.assertEqual(macros.defines, "$(LCSNoInternalFiberScheduler)")

    def test_sim_macros_emit_props_lines(self):
        m = self.reg.find("sim")
        macros = cb.build_macros([cb.BlockInstall(manifest=m)])
        self.assertIn("<LCSRoot>", macros.props_lines)
        self.assertIn("<LCSNoInternalFiberScheduler>", macros.props_lines)
        # 8 properties — 8 lines.
        self.assertEqual(len(macros.props_lines.split("\n")), 8)

    def test_sim_macros_split_libs_debug_release(self):
        m = self.reg.find("sim")
        macros = cb.build_macros([cb.BlockInstall(manifest=m)])
        self.assertIn(r"$(LCSLibDebug)\luisa-compute-solver-lib.lib", macros.debug_libs)
        self.assertIn(r"$(LCSLibRelease)\luisa-compute-solver-lib.lib", macros.release_libs)
        # Solver lib is config-tagged, so debug shouldn't contain the release path.
        self.assertNotIn(r"$(LCSLibRelease)", macros.debug_libs)
        self.assertNotIn(r"$(LCSLibDebug)", macros.release_libs)

    def test_video_macros_emit_mf_libs(self):
        m = self.reg.find("video")
        macros = cb.build_macros([cb.BlockInstall(manifest=m)])
        # All 6 link libs are universal → appear in both pipelines.
        for lib in ("d3d11.lib", "mf.lib", "mfplat.lib"):
            self.assertIn(lib, macros.debug_libs)
            self.assertIn(lib, macros.release_libs)

    def test_sim_video_macros_combined(self):
        macros = cb.build_macros(self._installs("sim", "video"))
        # Combined include paths still carry LCS macros (video contributes none).
        self.assertIn("$(LCSInclude)", macros.include_paths)
        # Combined debug libs carry both the solver lib AND mf libs.
        self.assertIn(r"$(LCSLibDebug)\luisa-compute-solver-lib.lib", macros.debug_libs)
        self.assertIn("mfplat.lib", macros.debug_libs)


class TestEngineAddonRegionBuilders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = _full_registry()

    def test_sim_vcxproj_clcompile_emits_absolute_path(self):
        out = cb.build_vcxproj_clcompile(
            [cb.BlockInstall(manifest=self.reg.find("sim"))]
        )
        self.assertIn(r'<ClCompile Include="..\src\newtype\physics\Physics.cpp" />', out)
        # The block-anchor prefix must NOT be prepended.
        self.assertNotIn("$(CinderBlocksDir)", out)
        self.assertNotIn("$(CinderBlocksLocalDir)", out)

    def test_sim_vcxproj_clinclude_emits_absolute_path(self):
        out = cb.build_vcxproj_clinclude(
            [cb.BlockInstall(manifest=self.reg.find("sim"))]
        )
        self.assertIn(r'<ClInclude Include="..\include\newtype\physics\Physics.h" />', out)

    def test_video_vcxproj_clcompile_has_3_sources(self):
        out = cb.build_vcxproj_clcompile(
            [cb.BlockInstall(manifest=self.reg.find("video"))]
        )
        self.assertEqual(out.count("<ClCompile "), 3)

    def test_sim_filters_use_source_files_filter_hierarchy(self):
        # Engine addons map absolute paths under Source Files\newtype\<area>
        # rather than Blocks\<folder>\..., matching the engine's existing
        # virtual folder layout.
        out = cb.build_filters_filter(
            [cb.BlockInstall(manifest=self.reg.find("sim"))]
        )
        self.assertIn(r'<Filter Include="Source Files\newtype\physics">', out)
        self.assertIn(r'<Filter Include="Header Files\newtype\physics">', out)
        # No Blocks\sim filter — engine addons don't contribute one.
        self.assertNotIn(r'<Filter Include="Blocks\sim">', out)

    def test_video_filters_use_media_subfilter(self):
        out = cb.build_filters_filter(
            [cb.BlockInstall(manifest=self.reg.find("video"))]
        )
        self.assertIn(r'<Filter Include="Source Files\newtype\media">', out)
        self.assertIn(r'<Filter Include="Header Files\newtype\media">', out)

    def test_sim_filters_assign_source_to_physics_filter(self):
        out = cb.build_filters_clcompile(
            [cb.BlockInstall(manifest=self.reg.find("sim"))]
        )
        self.assertIn(r"<Filter>Source Files\newtype\physics</Filter>", out)


class TestEngineAddonScaffoldDefines(unittest.TestCase):
    """The ;$(CinderBlocksDefines) suffix must be injected in all 3 configs."""

    def test_scaffold_injects_defines_macro_into_preprocessor_definitions(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "DefinesScaffold")
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            text = vcx.read_text(encoding="utf-8")
            # All 3 ItemDefinitionGroups have a PreprocessorDefinitions
            # element, and each carries the defines suffix.
            self.assertEqual(
                text.count(";$(CinderBlocksDefines)"),
                text.count("<PreprocessorDefinitions>"),
            )
            # The scaffold property region exposes the empty default.
            self.assertIn(
                "<CinderBlocksDefines></CinderBlocksDefines>",
                text,
            )

    def test_unscaffold_strips_defines_suffix(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "DefinesUnscaffold")
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            self.assertIn(";$(CinderBlocksDefines)", vcx.read_text(encoding="utf-8"))
            editor.unscaffold()
            self.assertNotIn(";$(CinderBlocksDefines)", vcx.read_text(encoding="utf-8"))
            self.assertNotIn("CinderBlocksDefines", vcx.read_text(encoding="utf-8"))


class TestEngineAddonPropsRegion(unittest.TestCase):
    """Scaffold adds the props properties region; round-trip preserves bytes."""

    def test_scaffold_creates_empty_props_region(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "PropsRegion")
            props = vcx.with_suffix(".props")
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            text = props.read_text(encoding="utf-8")
            self.assertIn(cb.begin_marker(cb.REGION_PROPS_PROPERTIES), text)
            self.assertIn(cb.end_marker(cb.REGION_PROPS_PROPERTIES), text)

    def test_unscaffold_removes_props_region(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "PropsRegionUn")
            props = vcx.with_suffix(".props")
            orig = props.read_bytes()
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            editor.unscaffold()
            self.assertEqual(props.read_bytes(), orig)


class TestCmdAddEngineAddons(unittest.TestCase):
    """End-to-end: `add sim` and `add video` against a generated project."""

    def test_add_sim_populates_props_region_with_lcs(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddSim")
            project = vcx.parent.parent
            # Construct args namespace with both --blocks-dir defaults.
            import argparse
            args = argparse.Namespace(
                blocks_dir=None,
                engine_addons_dir=None,
            )
            # Bypass CLI to inject our registry.
            blocks_dir = cb._resolve_blocks_dir(None)
            extras = [ENGINE_ADDONS_DIR]
            reg = cb.BlockRegistry(blocks_dir, extra_dirs=extras).scan()
            # Replicate cmd_add's state transitions on a controlled registry.
            state = cb.ProjectState(project, reg).maybe_load()
            state.ensure_sidecar(blocks_dir)
            closure = cb.resolve_deps(["sim"], reg)
            for folder, origin in closure:
                explicit = origin == "requested"
                rb = [] if explicit else [origin.split(":", 1)[1]]
                state.add_block(folder, mode=cb.MODE_REFERENCE, explicit=explicit, required_by=rb)
            state.save_sidecar()
            state.apply()

            props_text = vcx.with_suffix(".props").read_text(encoding="utf-8")
            # All 8 LCS properties lines are present inside the marker region.
            for tag in ("<LCSRoot>", "<LCSInclude>", "<LCSNoInternalFiberScheduler>"):
                self.assertIn(tag, props_text)
            # Region is populated.
            content = cb.get_region_content(props_text, cb.REGION_PROPS_PROPERTIES)
            self.assertIsNotNone(content)
            self.assertIn("<LCSRoot>", content)

    def test_add_sim_propagates_defines_to_vcxproj(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddSimDefines")
            project = vcx.parent.parent
            blocks_dir = cb._resolve_blocks_dir(None)
            reg = cb.BlockRegistry(blocks_dir, extra_dirs=[ENGINE_ADDONS_DIR]).scan()
            state = cb.ProjectState(project, reg).maybe_load()
            state.ensure_sidecar(blocks_dir)
            for folder, origin in cb.resolve_deps(["sim"], reg):
                state.add_block(folder, mode=cb.MODE_REFERENCE,
                                explicit=origin == "requested",
                                required_by=[] if origin == "requested" else [origin.split(":", 1)[1]])
            state.save_sidecar()
            state.apply()

            text = vcx.read_text(encoding="utf-8")
            # The defines macro value contains the LCS_NO_INTERNAL_FIBER_SCHEDULER reference.
            self.assertIn(
                "<CinderBlocksDefines>$(LCSNoInternalFiberScheduler)</CinderBlocksDefines>",
                text,
            )
            # The preprocessor element carries the suffix in all 3 configs.
            self.assertEqual(
                text.count(";$(CinderBlocksDefines)"),
                text.count("<PreprocessorDefinitions>"),
            )

    def test_add_video_links_mf_libs(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddVideo")
            project = vcx.parent.parent
            blocks_dir = cb._resolve_blocks_dir(None)
            reg = cb.BlockRegistry(blocks_dir, extra_dirs=[ENGINE_ADDONS_DIR]).scan()
            state = cb.ProjectState(project, reg).maybe_load()
            state.ensure_sidecar(blocks_dir)
            for folder, origin in cb.resolve_deps(["video"], reg):
                state.add_block(folder, mode=cb.MODE_REFERENCE,
                                explicit=origin == "requested",
                                required_by=[] if origin == "requested" else [origin.split(":", 1)[1]])
            state.save_sidecar()
            state.apply()

            text = vcx.read_text(encoding="utf-8")
            self.assertIn("mf.lib", text)
            self.assertIn("mfplat.lib", text)
            self.assertIn("d3d11.lib", text)
            # Source / header entries are absolute (no block anchor).
            self.assertIn(r'<ClCompile Include="..\src\newtype\media\VideoPlayer.cpp" />', text)
            self.assertIn(r'<ClInclude Include="..\include\newtype\media\VideoPlayer.h" />', text)

    def test_remove_sim_empties_props_region(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "RemoveSim")
            project = vcx.parent.parent
            blocks_dir = cb._resolve_blocks_dir(None)
            reg = cb.BlockRegistry(blocks_dir, extra_dirs=[ENGINE_ADDONS_DIR]).scan()
            state = cb.ProjectState(project, reg).maybe_load()
            state.ensure_sidecar(blocks_dir)
            for folder, origin in cb.resolve_deps(["sim"], reg):
                state.add_block(folder, mode=cb.MODE_REFERENCE,
                                explicit=origin == "requested",
                                required_by=[] if origin == "requested" else [origin.split(":", 1)[1]])
            state.save_sidecar()
            state.apply()

            # The marker region must contain the 8 LCS properties right after add.
            props_path = vcx.with_suffix(".props")
            content_before = cb.get_region_content(
                props_path.read_text(encoding="utf-8"), cb.REGION_PROPS_PROPERTIES
            )
            self.assertIn("<LCSRoot>", content_before)

            state.remove_block("sim")
            state.save_sidecar()
            state.apply()

            props_text = props_path.read_text(encoding="utf-8")
            content = cb.get_region_content(props_text, cb.REGION_PROPS_PROPERTIES)
            # Region still exists (scaffold markers stay) but content is empty.
            self.assertIsNotNone(content)
            self.assertEqual(content, "")
            # The vcxproj PreprocessorDefinitions no longer references the LCS define.
            vcx_text = vcx.read_text(encoding="utf-8")
            self.assertNotIn(
                "<CinderBlocksDefines>$(LCSNoInternalFiberScheduler)</CinderBlocksDefines>",
                vcx_text,
            )
            # generate_project.py now uses EngineTemplate.props (no LCS baked
            # in), so the marker region is the ONLY place <LCSRoot> could
            # appear. After remove, it must be globally absent.
            self.assertNotIn("<LCSRoot>", props_text)
            self.assertNotIn("LCS_NO_INTERNAL_FIBER_SCHEDULER", props_text)

    def test_full_round_trip_with_both_addons_byte_identity(self):
        # add sim + add video → remove both → unscaffold yields original bytes.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "FullRoundAddons")
            project = vcx.parent.parent
            orig_vcx = vcx.read_bytes()
            orig_props = vcx.with_suffix(".props").read_bytes()
            orig_filters = vcx.with_suffix(".vcxproj.filters").read_bytes()

            blocks_dir = cb._resolve_blocks_dir(None)
            reg = cb.BlockRegistry(blocks_dir, extra_dirs=[ENGINE_ADDONS_DIR]).scan()
            state = cb.ProjectState(project, reg).maybe_load()
            state.ensure_sidecar(blocks_dir)
            for folder, origin in cb.resolve_deps(["sim", "video"], reg):
                state.add_block(folder, mode=cb.MODE_REFERENCE,
                                explicit=origin == "requested",
                                required_by=[] if origin == "requested" else [origin.split(":", 1)[1]])
            state.save_sidecar()
            state.apply()

            # Now remove both and explicitly unscaffold.
            state.remove_block("sim")
            state.remove_block("video")
            state.save_sidecar()
            state.apply()

            editor = cb.ProjectEditor(vcx).load()
            editor.unscaffold()

            self.assertEqual(vcx.read_bytes(), orig_vcx)
            self.assertEqual(vcx.with_suffix(".props").read_bytes(), orig_props)
            self.assertEqual(
                vcx.with_suffix(".vcxproj.filters").read_bytes(),
                orig_filters,
            )


# ===========================================================================
# P7: DirectML addon — <additionalOption>, <globalsProperty>, runtimeobject
# ===========================================================================


class TestDirectMLParser(unittest.TestCase):
    """Verify parser handles <additionalOption> and <globalsProperty>."""

    @classmethod
    def setUpClass(cls):
        cls.reg = _full_registry()

    def test_directml_parses_runtimeobject_lib(self):
        m = self.reg.find("directml")
        self.assertEqual(len(m.static_libs), 1)
        lib = m.static_libs[0]
        self.assertEqual(lib.path, "runtimeobject.lib")
        self.assertTrue(lib.absolute)
        # Universal — no config tag, applies to Debug + Debug_Runtime + Release.
        self.assertIsNone(lib.config)

    def test_directml_parses_additional_option(self):
        m = self.reg.find("directml")
        self.assertEqual(m.additional_options, ["/bigobj"])

    def test_directml_parses_globals_property(self):
        m = self.reg.find("directml")
        self.assertEqual(len(m.globals_properties), 1)
        gp = m.globals_properties[0]
        self.assertEqual(gp.name, "CppWinRTEnabled")
        self.assertEqual(gp.value, "true")

    def test_directml_has_no_sources_headers_includes(self):
        # The addon is pure compile/link/build-settings wiring — no in-engine
        # code. Document that assumption so accidental additions are caught.
        m = self.reg.find("directml")
        self.assertEqual(m.sources, [])
        self.assertEqual(m.headers, [])
        self.assertEqual(m.include_paths, [])
        self.assertEqual(m.properties, [])  # No .props property declarations


class TestDirectMLMacroBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = _full_registry()

    def test_runtimeobject_in_both_lib_pipelines(self):
        m = self.reg.find("directml")
        macros = cb.build_macros([cb.BlockInstall(manifest=m)])
        self.assertIn("runtimeobject.lib", macros.debug_libs)
        self.assertIn("runtimeobject.lib", macros.release_libs)

    def test_additional_options_joined_by_space(self):
        m = self.reg.find("directml")
        macros = cb.build_macros([cb.BlockInstall(manifest=m)])
        self.assertEqual(macros.additional_options, "/bigobj")

    def test_globals_lines_emit_cppwinrt_enabled(self):
        m = self.reg.find("directml")
        macros = cb.build_macros([cb.BlockInstall(manifest=m)])
        self.assertIn("<CppWinRTEnabled>true</CppWinRTEnabled>", macros.globals_lines)


class TestAdditionalOptionsScaffold(unittest.TestCase):
    """The $(CinderBlocksAdditionalOptions) suffix must be in all 3 configs."""

    def test_scaffold_injects_additional_options_macro(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddOptScaffold")
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            text = vcx.read_text(encoding="utf-8")
            # All 3 ItemDefinitionGroups have an <AdditionalOptions> element;
            # each carries the macro reference (with leading space).
            self.assertEqual(
                text.count(" $(CinderBlocksAdditionalOptions)"),
                text.count("<AdditionalOptions>"),
            )
            # The scaffold property exposes the empty default.
            self.assertIn(
                "<CinderBlocksAdditionalOptions></CinderBlocksAdditionalOptions>",
                text,
            )

    def test_unscaffold_strips_additional_options_suffix(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddOptUn")
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            self.assertIn(" $(CinderBlocksAdditionalOptions)", vcx.read_text(encoding="utf-8"))
            editor.unscaffold()
            self.assertNotIn("CinderBlocksAdditionalOptions", vcx.read_text(encoding="utf-8"))


class TestGlobalsRegionScaffold(unittest.TestCase):
    """REGION_VCXPROJ_GLOBALS sits inside <PropertyGroup Label='Globals'>."""

    def test_scaffold_creates_empty_globals_region(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "GlobalsScaffold")
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            text = vcx.read_text(encoding="utf-8")
            self.assertIn(cb.begin_marker(cb.REGION_VCXPROJ_GLOBALS), text)
            self.assertIn(cb.end_marker(cb.REGION_VCXPROJ_GLOBALS), text)
            # The region sits inside the Globals PropertyGroup: the BEGIN
            # marker must appear after <Keyword>Win32Proj</Keyword> and the
            # END marker must appear before </PropertyGroup> that closes Globals.
            keyword_idx = text.find("<Keyword>Win32Proj</Keyword>")
            begin_idx = text.find(cb.begin_marker(cb.REGION_VCXPROJ_GLOBALS))
            self.assertGreater(begin_idx, keyword_idx)

    def test_unscaffold_removes_globals_region(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "GlobalsUn")
            editor = cb.ProjectEditor(vcx).load()
            editor.scaffold()
            self.assertIn(cb.begin_marker(cb.REGION_VCXPROJ_GLOBALS),
                          vcx.read_text(encoding="utf-8"))
            editor.unscaffold()
            self.assertNotIn(cb.REGION_VCXPROJ_GLOBALS,
                             vcx.read_text(encoding="utf-8"))


class TestCmdAddDirectML(unittest.TestCase):
    """End-to-end: `add directml` against a freshly-generated project."""

    def _add_directml(self, project_dir, vcx):
        blocks_dir = cb._resolve_blocks_dir(None)
        reg = cb.BlockRegistry(blocks_dir, extra_dirs=[ENGINE_ADDONS_DIR]).scan()
        state = cb.ProjectState(project_dir, reg).maybe_load()
        state.ensure_sidecar(blocks_dir)
        for folder, origin in cb.resolve_deps(["directml"], reg):
            state.add_block(folder, mode=cb.MODE_REFERENCE,
                            explicit=origin == "requested",
                            required_by=[] if origin == "requested" else [origin.split(":", 1)[1]])
        state.save_sidecar()
        state.apply()

    def test_add_directml_links_runtimeobject(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddDml")
            self._add_directml(vcx.parent.parent, vcx)
            text = vcx.read_text(encoding="utf-8")
            # Universal lib: appears in both debug and release lib macros.
            self.assertIn(
                "<CinderBlocksDebugLibs>runtimeobject.lib</CinderBlocksDebugLibs>",
                text,
            )
            self.assertIn(
                "<CinderBlocksReleaseLibs>runtimeobject.lib</CinderBlocksReleaseLibs>",
                text,
            )

    def test_add_directml_compiles_with_bigobj(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddDmlBigobj")
            self._add_directml(vcx.parent.parent, vcx)
            text = vcx.read_text(encoding="utf-8")
            self.assertIn(
                "<CinderBlocksAdditionalOptions>/bigobj</CinderBlocksAdditionalOptions>",
                text,
            )
            # Suffix injected into <AdditionalOptions> of all 3 configs.
            self.assertEqual(
                text.count(" $(CinderBlocksAdditionalOptions)"),
                text.count("<AdditionalOptions>"),
            )

    def test_add_directml_sets_cppwinrt_enabled_in_globals(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddDmlGlobals")
            self._add_directml(vcx.parent.parent, vcx)
            text = vcx.read_text(encoding="utf-8")
            self.assertIn("<CppWinRTEnabled>true</CppWinRTEnabled>", text)
            # The flag sits inside the marker region, which sits inside the
            # Globals PropertyGroup (after <Keyword>Win32Proj</Keyword>).
            keyword_idx = text.find("<Keyword>Win32Proj</Keyword>")
            flag_idx = text.find("<CppWinRTEnabled>true</CppWinRTEnabled>")
            self.assertGreater(flag_idx, keyword_idx)

    def test_remove_directml_clears_all_three_wirings(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "RemoveDml")
            project = vcx.parent.parent
            self._add_directml(project, vcx)
            text_before = vcx.read_text(encoding="utf-8")
            self.assertIn("runtimeobject.lib", text_before)
            self.assertIn("/bigobj", text_before)
            self.assertIn("<CppWinRTEnabled>true</CppWinRTEnabled>", text_before)

            # Remove via state flow.
            blocks_dir = cb._resolve_blocks_dir(None)
            reg = cb.BlockRegistry(blocks_dir, extra_dirs=[ENGINE_ADDONS_DIR]).scan()
            state = cb.ProjectState(project, reg).maybe_load()
            state.load()
            state.remove_block("directml")
            state.save_sidecar()
            state.apply()

            text = vcx.read_text(encoding="utf-8")
            self.assertNotIn("runtimeobject.lib", text)
            self.assertNotIn("/bigobj", text)
            self.assertNotIn("<CppWinRTEnabled>true</CppWinRTEnabled>", text)
            # Region markers stay (scaffold still up); content empty.
            content = cb.get_region_content(text, cb.REGION_VCXPROJ_GLOBALS)
            self.assertIsNotNone(content)
            self.assertEqual(content, "")

    def test_full_round_trip_byte_identity_with_directml(self):
        # add directml → remove → unscaffold yields original bytes.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "FullRoundDml")
            project = vcx.parent.parent
            orig_vcx = vcx.read_bytes()
            orig_props = vcx.with_suffix(".props").read_bytes()
            orig_filters = vcx.with_suffix(".vcxproj.filters").read_bytes()

            self._add_directml(project, vcx)

            blocks_dir = cb._resolve_blocks_dir(None)
            reg = cb.BlockRegistry(blocks_dir, extra_dirs=[ENGINE_ADDONS_DIR]).scan()
            state = cb.ProjectState(project, reg).maybe_load()
            state.load()
            state.remove_block("directml")
            state.save_sidecar()
            state.apply()

            editor = cb.ProjectEditor(vcx).load()
            editor.unscaffold()

            self.assertEqual(vcx.read_bytes(), orig_vcx)
            self.assertEqual(vcx.with_suffix(".props").read_bytes(), orig_props)
            self.assertEqual(
                vcx.with_suffix(".vcxproj.filters").read_bytes(),
                orig_filters,
            )


if __name__ == "__main__":
    # When run as a script, default to verbose; allow -q to override.
    import argparse as _ap

    _p = _ap.ArgumentParser()
    _p.add_argument("-q", "--quiet", action="store_true")
    _q, _r = _p.parse_known_args()
    unittest.main(verbosity=0 if _q else 2, argv=[sys.argv[0]] + _r)


# ===========================================================================
# P3: MacroBuilder, region content, ProjectState, cmd_add/remove/update
# ===========================================================================


def _manifest(folder: str) -> cb.BlockManifest:
    """Look up a real manifest from the registry (used by P3 unit tests)."""
    reg = cb.BlockRegistry(BLOCKS_DIR).scan()
    m = reg.find(folder)
    assert m is not None, f"Block not found: {folder}"
    return m


def _install(folder: str, mode: str = cb.MODE_REFERENCE) -> cb.BlockInstall:
    """Wrap a real manifest in a BlockInstall (default reference mode)."""
    return cb.BlockInstall(manifest=_manifest(folder), mode=mode)


class TestMacroBuilder(unittest.TestCase):
    def test_osc_yields_only_include_paths(self):
        # OSC has <includePath>src</includePath> and no libs/buildCopy.
        macros = cb.build_macros([_install("OSC")])
        self.assertEqual(macros.include_paths, r"$(CinderBlocksDir)\OSC\src")
        self.assertEqual(macros.lib_paths, "")
        self.assertEqual(macros.debug_libs, "")
        self.assertEqual(macros.release_libs, "")
        self.assertEqual(macros.post_build, "")

    def test_cinder_ndi_yields_lib_and_postbuild(self):
        m = _manifest("Cinder-NDI")
        macros = cb.build_macros([cb.BlockInstall(manifest=m, mode=cb.MODE_REFERENCE)])
        # NDI has includePath 'include' + 'lib/NDI/include', one static lib
        # (universal, no config), one buildCopy DLL.
        self.assertIn(r"$(CinderBlocksDir)\Cinder-NDI\include", macros.include_paths)
        self.assertIn(
            r"$(CinderBlocksDir)\Cinder-NDI\lib\NDI\include", macros.include_paths
        )
        self.assertIn(
            r"$(CinderBlocksDir)\Cinder-NDI\lib\NDI\lib\$(PlatformTarget)\Processing.NDI.Lib.$(PlatformTarget).lib",
            macros.debug_libs,
        )
        # Universal lib => appears in both Debug and Release.
        self.assertEqual(macros.debug_libs, macros.release_libs)
        # PostBuild has the xcopy chain.
        self.assertIn(
            'xcopy /y "$(CinderBlocksDir)\\Cinder-NDI\\lib\\NDI\\bin\\$(PlatformTarget)\\Processing.NDI.Lib.$(PlatformTarget).dll" "$(OutDir)\\"',
            macros.post_build,
        )

    def test_cinder_urg_splits_debug_release(self):
        m = _manifest("Cinder-URG")
        macros = cb.build_macros([cb.BlockInstall(manifest=m, mode=cb.MODE_REFERENCE)])
        # setupapi.lib is universal; urg_cpp_debug.lib is debug-only,
        # urg_cpp.lib is release-only.
        self.assertIn("setupapi.lib", macros.debug_libs)
        self.assertIn("setupapi.lib", macros.release_libs)
        self.assertIn("urg_cpp_debug.lib", macros.debug_libs)
        self.assertNotIn("urg_cpp_debug.lib", macros.release_libs)
        self.assertIn("urg_cpp.lib", macros.release_libs)
        self.assertNotIn("urg_cpp.lib", macros.debug_libs)

    def test_absolute_static_lib_emitted_verbatim(self):
        # Cairo has <staticLibrary absolute="true">libpng.lib</staticLibrary>.
        m = _manifest("Cairo")
        macros = cb.build_macros([cb.BlockInstall(manifest=m, mode=cb.MODE_REFERENCE)])
        self.assertIn("libpng.lib", macros.debug_libs)
        self.assertIn("libpng.lib", macros.release_libs)
        # The absolute path should NOT be wrapped in $(CinderBlocksDir).
        self.assertNotIn("$(CinderBlocksDir)\\Cairo\\libpng.lib", macros.debug_libs)

    def test_cinder_attribute_include_path_uses_cinder_root(self):
        # Cinder-WebSocketPP: <includePath cinder="true">include/asio</includePath>
        m = _manifest("Cinder-WebSocketPP")
        macros = cb.build_macros([cb.BlockInstall(manifest=m, mode=cb.MODE_REFERENCE)])
        self.assertIn(r"$(CinderRoot)\include\asio", macros.include_paths)
        # The block-local path uses $(CinderBlocksDir) as usual.
        self.assertIn(r"$(CinderBlocksDir)\Cinder-WebSocketPP\src", macros.include_paths)

    def test_multiple_blocks_concatenate(self):
        macros = cb.build_macros([_install("OSC"), _install("Cinder-NDI")])
        # Both block-local include paths appear, OSC first then NDI (input order).
        self.assertEqual(
            macros.include_paths,
            r"$(CinderBlocksDir)\OSC\src"
            ";"
            r"$(CinderBlocksDir)\Cinder-NDI\include"
            ";"
            r"$(CinderBlocksDir)\Cinder-NDI\lib\NDI\include",
        )


class TestRegionContentBuilders(unittest.TestCase):
    def test_vcxproj_clcompile_for_osc(self):
        out = cb.build_vcxproj_clcompile([_install("OSC")])
        self.assertIn(
            r'<ClCompile Include="$(CinderBlocksDir)\OSC\src\cinder\osc\Osc.cpp" />',
            out,
        )

    def test_vcxproj_clinclude_for_osc(self):
        out = cb.build_vcxproj_clinclude([_install("OSC")])
        self.assertIn(
            r'<ClInclude Include="$(CinderBlocksDir)\OSC\src\cinder\osc\Osc.h" />',
            out,
        )

    def test_empty_manifest_list_yields_empty_itemgroup(self):
        out = cb.build_vcxproj_clcompile([])
        self.assertIn("<ItemGroup>", out)
        # No ClCompile entries.
        self.assertNotIn("ClCompile", out)

    def test_filters_creates_per_block_filter_definition(self):
        out = cb.build_filters_filter([_install("OSC")])
        # Top-level Blocks filter sits alongside Source Files / Header Files.
        self.assertIn(r'<Filter Include="Blocks">', out)
        self.assertIn(r'<Filter Include="Blocks\OSC">', out)
        # Nested source subdir filter should also be defined.
        self.assertIn(r'<Filter Include="Blocks\OSC\src\cinder\osc">', out)
        # Intermediate parents should be present too.
        self.assertIn(r'<Filter Include="Blocks\OSC\src\cinder">', out)

    def test_filters_src_include_split_for_ndi(self):
        # Cinder-NDI has files in src/ (ClCompile) and include/ (ClInclude).
        # The filter structure must reflect both subdirs separately, matching
        # TinderBox's Blocks\<folder>\{src,include} layout.
        inst = _install("Cinder-NDI")
        filters = cb.build_filters_filter([inst])
        self.assertIn(r'<Filter Include="Blocks\Cinder-NDI\src">', filters)
        self.assertIn(r'<Filter Include="Blocks\Cinder-NDI\include">', filters)
        clcompile = cb.build_filters_clcompile([inst])
        self.assertIn(r"<Filter>Blocks\Cinder-NDI\src</Filter>", clcompile)
        clinclude = cb.build_filters_clinclude([inst])
        self.assertIn(r"<Filter>Blocks\Cinder-NDI\include</Filter>", clinclude)

    def test_filters_no_blocks_filter_when_no_manifests(self):
        out = cb.build_filters_filter([])
        self.assertNotIn(r'<Filter Include="Blocks">', out)

    def test_filters_assigns_files_to_block_filters(self):
        out = cb.build_filters_clcompile([_install("OSC")])
        self.assertIn(r"<Filter>Blocks\OSC\src\cinder\osc</Filter>", out)
        out_h = cb.build_filters_clinclude([_install("OSC")])
        self.assertIn(r"<Filter>Blocks\OSC\src\cinder\osc</Filter>", out_h)

    def test_filter_guids_are_deterministic(self):
        # Same block -> same GUID on every call.
        a = cb.build_filters_filter([_install("OSC")])
        b = cb.build_filters_filter([_install("OSC")])
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# ProjectState + sidecar round-trip
# ---------------------------------------------------------------------------


class TestSidecarIO(unittest.TestCase):
    def test_init_then_save_then_load_round_trips(self):
        with _SameDriveTempDir() as tmp:
            proj = tmp / "P"
            (proj / "vc2022").mkdir(parents=True)
            data = cb.init_sidecar(BLOCKS_DIR)
            data["blocks"]["OSC"] = {
                "block_id": "org.libcinder.osc",
                "mode": "reference",
                "explicit": True,
                "required_by": [],
            }
            cb.save_sidecar(proj, data)
            loaded = cb.load_sidecar(proj)
            self.assertEqual(loaded, data)


# ---------------------------------------------------------------------------
# cmd_add / cmd_remove / cmd_update end-to-end on a generated project
# ---------------------------------------------------------------------------


def _run_cli(*argv: str) -> int:
    """Invoke the CLI with the given args. Returns the exit code."""
    return cb.main(list(argv))


class TestCmdAddRemoveUpdate(unittest.TestCase):
    """Drive the full CLI against freshly-generated projects."""

    def test_add_osc_populates_vcxproj_regions(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddOsc")
            project = vcx.parent.parent  # vc2022/.. = project root
            self.assertEqual(_run_cli("add", "OSC", "--project", str(project)), 0)

            text = vcx.read_text(encoding="utf-8")
            # Macro value populated.
            self.assertIn(
                r"<CinderBlocksIncludePaths>$(CinderBlocksDir)\OSC\src</CinderBlocksIncludePaths>",
                text,
            )
            # ClCompile / ClInclude entries present.
            self.assertIn(
                r'<ClCompile Include="$(CinderBlocksDir)\OSC\src\cinder\osc\Osc.cpp" />',
                text,
            )
            self.assertIn(
                r'<ClInclude Include="$(CinderBlocksDir)\OSC\src\cinder\osc\Osc.h" />',
                text,
            )

            # Sidecar JSON present with the expected entry.
            sidecar = cb.load_sidecar(project)
            self.assertIn("OSC", sidecar["blocks"])
            self.assertTrue(sidecar["blocks"]["OSC"]["explicit"])

    def test_add_tuio_auto_installs_osc_dependency(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddTuio")
            project = vcx.parent.parent
            self.assertEqual(_run_cli("add", "TUIO", "--project", str(project)), 0)

            sidecar = cb.load_sidecar(project)
            # Both installed, TUIO explicit, OSC auto.
            self.assertIn("TUIO", sidecar["blocks"])
            self.assertIn("OSC", sidecar["blocks"])
            self.assertTrue(sidecar["blocks"]["TUIO"]["explicit"])
            self.assertFalse(sidecar["blocks"]["OSC"]["explicit"])
            self.assertEqual(sidecar["blocks"]["OSC"]["required_by"], ["TUIO"])

    def test_add_is_idempotent(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "AddIdem")
            project = vcx.parent.parent
            self.assertEqual(_run_cli("add", "OSC", "--project", str(project)), 0)
            after_first = vcx.read_bytes()
            # Second add should be a no-op for the project files.
            _run_cli("add", "OSC", "--project", str(project))
            self.assertEqual(vcx.read_bytes(), after_first)

    def test_add_dry_run_does_not_write(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "DryRun")
            project = vcx.parent.parent
            original = vcx.read_bytes()
            self.assertEqual(
                _run_cli("add", "OSC", "--project", str(project), "--dry-run"),
                0,
            )
            self.assertEqual(vcx.read_bytes(), original)
            # Sidecar should not have been created either.
            self.assertIsNone(cb.load_sidecar(project))

    def test_remove_keeps_scaffold_but_empties_regions(self):
        # Removing the only block leaves the scaffold markers in place
        # (cheap for the next add), but empties the regions and macros.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "RemoveKeep")
            project = vcx.parent.parent
            _run_cli("add", "OSC", "--project", str(project))
            self.assertEqual(_run_cli("remove", "OSC", "--project", str(project)), 0)

            text = vcx.read_text(encoding="utf-8")
            # Scaffold markers still present.
            self.assertIn(cb.begin_marker(cb.REGION_PROPERTIES), text)
            # OSC entries gone.
            self.assertNotIn(r"$(CinderBlocksDir)\OSC", text)
            # Macro value empty again.
            self.assertIn(
                "<CinderBlocksIncludePaths></CinderBlocksIncludePaths>", text
            )
            # Sidecar still exists but blocks dict is empty.
            sidecar = cb.load_sidecar(project)
            self.assertEqual(sidecar["blocks"], {})

    def test_remove_then_unscaffold_restores_byte_identity(self):
        # Full round-trip: add -> remove -> unscaffold = original bytes.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "FullRound")
            project = vcx.parent.parent
            orig_vcx = vcx.read_bytes()
            orig_filters = vcx.with_suffix(".vcxproj.filters").read_bytes()
            orig_props = vcx.with_suffix(".props").read_bytes()

            _run_cli("--blocks-dir", str(BLOCKS_DIR),
                     "add", "Cinder-NDI", "OSC", "--project", str(project))
            _run_cli("--blocks-dir", str(BLOCKS_DIR),
                     "remove", "Cinder-NDI", "OSC", "--project", str(project))

            # Explicit unscaffold to strip the dormant markers.
            editor = cb.ProjectEditor(vcx).load()
            self.assertTrue(editor.unscaffold())

            self.assertEqual(vcx.read_bytes(), orig_vcx)
            self.assertEqual(
                vcx.with_suffix(".vcxproj.filters").read_bytes(),
                orig_filters,
            )
            self.assertEqual(vcx.with_suffix(".props").read_bytes(), orig_props)

    def test_remove_refuses_when_required_by_others(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "RefuseRemove")
            project = vcx.parent.parent
            _run_cli("add", "TUIO", "--project", str(project))
            # OSC is now required by TUIO; remove should refuse.
            rc = _run_cli("remove", "OSC", "--project", str(project))
            self.assertNotEqual(rc, 0)
            # OSC should still be installed.
            sidecar = cb.load_sidecar(project)
            self.assertIn("OSC", sidecar["blocks"])

    def test_remove_cascade_strips_orphaned_dependency(self):
        # remove TUIO -> OSC was auto-pulled-in and should be removed too.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "Cascade")
            project = vcx.parent.parent
            _run_cli("add", "TUIO", "--project", str(project))
            self.assertEqual(_run_cli("remove", "TUIO", "--project", str(project)), 0)

            sidecar = cb.load_sidecar(project)
            # Both should be gone.
            self.assertEqual(sidecar["blocks"], {})
            # vcxproj should no longer reference either block.
            text = vcx.read_text(encoding="utf-8")
            self.assertNotIn(r"$(CinderBlocksDir)\TUIO", text)
            self.assertNotIn(r"$(CinderBlocksDir)\OSC", text)

    def test_remove_keep_deps_does_not_cascade(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "KeepDeps")
            project = vcx.parent.parent
            _run_cli("add", "TUIO", "--project", str(project))
            self.assertEqual(
                _run_cli("remove", "TUIO", "--project", str(project), "--keep-deps"),
                0,
            )
            sidecar = cb.load_sidecar(project)
            # TUIO gone but OSC preserved as orphan (still implicit).
            self.assertNotIn("TUIO", sidecar["blocks"])
            self.assertIn("OSC", sidecar["blocks"])
            self.assertEqual(sidecar["blocks"]["OSC"]["required_by"], [])

    def test_update_is_noop_when_unchanged(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "UpdateNoop")
            project = vcx.parent.parent
            _run_cli("add", "OSC", "--project", str(project))
            # Re-running update should report "already in sync".
            rc = _run_cli("update", "--project", str(project))
            self.assertEqual(rc, 0)


# ===========================================================================
# P3.1: Multi-block sequential adds (regression coverage)
# ===========================================================================
#
# The user reported that adding multiple blocks back-to-back via separate
# `add` invocations can corrupt the vcxproj. These tests exercise that flow
# with increasingly tricky orderings (with deps, without deps, mixed modes,
# re-adding after a remove). If any one fails, the multi-block bug is back.


def _count_region(text: str, region: str) -> int:
    """Count region BEGIN markers — should always be 0 or 1 per region name."""
    return text.count(cb.begin_marker(region))


def _assert_vcproj_well_formed(testcase: unittest.TestCase, vcx_path: Path):
    """Catch corruption: every scaffold region appears exactly once, the
    file parses as XML, and no ItemGroup has a stray begin/end marker inside
    its body (which would mean a previous add spliced content into the wrong
    place)."""
    text = vcx_path.read_text(encoding="utf-8")
    for region in (
        cb.REGION_PROPERTIES,
        cb.REGION_CLCOMPILE,
        cb.REGION_CLINCLUDE,
        cb.REGION_VCXPROJ_GLOBALS,
    ):
        testcase.assertEqual(
            _count_region(text, region), 1,
            f"region {region!r} should appear exactly once"
        )
        testcase.assertEqual(
            text.count(cb.end_marker(region)), 1,
            f"region {region!r} END should appear exactly once"
        )
    # The four scaffold regions must not nest or overlap: their BEGIN/END
    # markers must alternate properly. Easiest check — strip every region
    # and confirm the result equals the un-scaffolded bytes the original
    # generate_project.py would have produced. We instead sanity-check via
    # XML parsing; a corrupted structure throws ParseError.
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(text)
    except ET.ParseError as e:
        testcase.fail(f"vcxproj is not well-formed XML: {e}")

    # Filters file too.
    filters_path = vcx_path.with_suffix(".vcxproj.filters")
    flt_text = filters_path.read_text(encoding="utf-8")
    for region in (
        cb.REGION_FILTERS_FILTER,
        cb.REGION_FILTERS_CLCOMPILE,
        cb.REGION_FILTERS_CLINCLUDE,
    ):
        testcase.assertEqual(_count_region(flt_text, region), 1,
                             f"filters region {region!r} missing or duplicated")
    try:
        ET.fromstring(flt_text)
    except ET.ParseError as e:
        testcase.fail(f"filters file is not well-formed XML: {e}")


# Blocks present in BOTH Cinder\blocks and CinderHead\blocks so these tests
# pass under either default. Picked for diverse manifest shapes:
#   OSC     — has sources + include path, no libs
#   Cairo   — has absolute static lib, single source/header at top level
#   TUIO    — pulls OSC as a dep (so a sequential add of TUIO after OSC
#             exercises the "dep already installed" merge path)
#   Box2D   — heavy multi-pattern glob source list (stress on filters)
#   Clipper — single source/header, typo in includePath (parser edge case)
SEQ_ADD_BLOCKS_CINDER = ["OSC", "Cairo", "TUIO", "Box2D", "Clipper"]


class TestSequentialAdds(unittest.TestCase):
    """Add blocks in separate `add` invocations; verify structure stays clean."""

    def test_add_three_blocks_separately(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "SeqThree")
            project = vcx.parent.parent
            self.assertEqual(_run_cli("add", "OSC", "--project", str(project)), 0)
            _assert_vcproj_well_formed(self, vcx)
            self.assertEqual(_run_cli("add", "Cairo", "--project", str(project)), 0)
            _assert_vcproj_well_formed(self, vcx)
            self.assertEqual(_run_cli("add", "Box2D", "--project", str(project)), 0)
            _assert_vcproj_well_formed(self, vcx)

            # All three blocks present in sidecar.
            sidecar = cb.load_sidecar(project)
            for folder in ("OSC", "Cairo", "Box2D"):
                self.assertIn(folder, sidecar["blocks"])

            # Each block contributes at least one ClCompile entry to the vcxproj.
            text = vcx.read_text(encoding="utf-8")
            self.assertIn(r"$(CinderBlocksDir)\OSC\src\cinder\osc\Osc.cpp", text)
            self.assertIn(r"$(CinderBlocksDir)\Cairo\src\Cairo.cpp", text)

    def test_add_dependency_after_its_requester(self):
        # Adding OSC then TUIO: TUIO requires OSC, so OSC is already installed
        # when TUIO is added. The required_by merge must not duplicate and
        # must reflect the real dependency edge (TUIO requires OSC) even
        # though OSC was pulled in explicitly first — cascade removal needs
        # accurate required_by edges to reason correctly.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "SeqDepAfter")
            project = vcx.parent.parent
            self.assertEqual(_run_cli("add", "OSC", "--project", str(project)), 0)
            _assert_vcproj_well_formed(self, vcx)
            self.assertEqual(_run_cli("add", "TUIO", "--project", str(project)), 0)
            _assert_vcproj_well_formed(self, vcx)

            sidecar = cb.load_sidecar(project)
            # Both stay explicit (OSC from first add, TUIO from second).
            self.assertTrue(sidecar["blocks"]["OSC"]["explicit"])
            self.assertTrue(sidecar["blocks"]["TUIO"]["explicit"])
            # OSC's required_by picks up TUIO (the merge is union, not
            # clobber) — without this, removing TUIO couldn't tell you OSC
            # was being depended on.
            self.assertEqual(sidecar["blocks"]["OSC"]["required_by"], ["TUIO"])

            text = vcx.read_text(encoding="utf-8")
            # Exactly one entry for each (no duplicates from the merge).
            self.assertEqual(text.count(r"$(CinderBlocksDir)\OSC\src\cinder\osc\Osc.cpp"), 1)
            self.assertEqual(text.count(r"$(CinderBlocksDir)\TUIO\src\cinder\tuio\Tuio.cpp"), 1)

    def test_add_requester_then_dependency_already_present(self):
        # TUIO first (auto-pulls OSC), then explicit add of OSC.
        # OSC must remain listed and switch to explicit (or stay implicit —
        # either is reasonable, but the tool shouldn't crash or duplicate).
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "SeqRequesterFirst")
            project = vcx.parent.parent
            self.assertEqual(_run_cli("add", "TUIO", "--project", str(project)), 0)
            _assert_vcproj_well_formed(self, vcx)
            self.assertEqual(_run_cli("add", "OSC", "--project", str(project)), 0)
            _assert_vcproj_well_formed(self, vcx)

            sidecar = cb.load_sidecar(project)
            # OSC is now explicit (user asked for it directly).
            self.assertTrue(sidecar["blocks"]["OSC"]["explicit"])
            # TUIO still references OSC in its required_by.
            self.assertIn("TUIO", sidecar["blocks"]["OSC"]["required_by"])

    def test_five_blocks_back_to_back(self):
        # Stress test: many sequential adds, including a dep chain.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "SeqFive")
            project = vcx.parent.parent
            for name in SEQ_ADD_BLOCKS_CINDER:
                self.assertEqual(_run_cli("add", name, "--project", str(project)), 0)
                _assert_vcproj_well_formed(self, vcx)

            sidecar = cb.load_sidecar(project)
            # All five folders (or their auto-pulled deps) should be present.
            for folder in SEQ_ADD_BLOCKS_CINDER:
                self.assertIn(folder, sidecar["blocks"])

    def test_add_then_remove_then_add(self):
        # add OSC, remove OSC, add OSC again — the re-add must repopulate
        # the regions without leaving stale markers from the remove.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "SeqReadd")
            project = vcx.parent.parent
            self.assertEqual(_run_cli("add", "OSC", "--project", str(project)), 0)
            self.assertEqual(_run_cli("remove", "OSC", "--project", str(project)), 0)
            self.assertEqual(_run_cli("add", "OSC", "--project", str(project)), 0)
            _assert_vcproj_well_formed(self, vcx)

            text = vcx.read_text(encoding="utf-8")
            self.assertIn(r"$(CinderBlocksDir)\OSC\src\cinder\osc\Osc.cpp", text)

    def test_single_call_multi_block_equals_sequential(self):
        # Adding 3 blocks in one call must produce the same vcxproj bytes as
        # adding them in three separate calls (the apply() idempotency
        # guarantee). Catches subtle ordering bugs that only manifest when
        # the sidecar is built up incrementally.
        #
        # Both projects use the same name (different tmp dirs) so their
        # generate_project.py-emitted ProjectGuids match — only the
        # cinder-blocks regions differ between them, which is what we want
        # to compare.
        with _SameDriveTempDir() as tmp_a, _SameDriveTempDir() as tmp_b:
            vcx_a = _generate_project(tmp_a, "Batch")
            vcx_b = _generate_project(tmp_b, "Batch")
            project_a = vcx_a.parent.parent
            project_b = vcx_b.parent.parent

            # Sanity: the freshly-generated bytes match (same name → same GUID).
            self.assertEqual(vcx_a.read_bytes(), vcx_b.read_bytes())

            # Batch A: one call with three blocks
            self.assertEqual(_run_cli(
                "add", "OSC", "Cairo", "Box2D",
                "--project", str(project_a)
            ), 0)

            # Batch B: three sequential calls
            self.assertEqual(_run_cli("add", "OSC", "--project", str(project_b)), 0)
            self.assertEqual(_run_cli("add", "Cairo", "--project", str(project_b)), 0)
            self.assertEqual(_run_cli("add", "Box2D", "--project", str(project_b)), 0)

            text_a = vcx_a.read_text(encoding="utf-8")
            text_b = vcx_b.read_text(encoding="utf-8")
            # Region contents are derived deterministically from the sidecar
            # (sorted by folder), so both should produce identical bytes.
            self.assertEqual(text_a, text_b)
            # Same for filters.
            self.assertEqual(
                vcx_a.with_suffix(".vcxproj.filters").read_text(encoding="utf-8"),
                vcx_b.with_suffix(".vcxproj.filters").read_text(encoding="utf-8"),
            )

    def test_full_round_trip_after_sequential_adds(self):
        # Add 3 blocks sequentially, then remove all 3, then unscaffold —
        # the project must return to its original generate_project.py bytes.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "SeqFullRound")
            project = vcx.parent.parent
            orig_vcx = vcx.read_bytes()
            orig_filters = vcx.with_suffix(".vcxproj.filters").read_bytes()
            orig_props = vcx.with_suffix(".props").read_bytes()

            self.assertEqual(_run_cli("add", "OSC", "--project", str(project)), 0)
            self.assertEqual(_run_cli("add", "Cairo", "--project", str(project)), 0)
            self.assertEqual(_run_cli("add", "Box2D", "--project", str(project)), 0)
            self.assertEqual(_run_cli(
                "remove", "OSC", "Cairo", "Box2D", "--project", str(project)
            ), 0)

            editor = cb.ProjectEditor(vcx).load()
            self.assertTrue(editor.unscaffold())

            self.assertEqual(vcx.read_bytes(), orig_vcx)
            self.assertEqual(
                vcx.with_suffix(".vcxproj.filters").read_bytes(), orig_filters
            )
            self.assertEqual(vcx.with_suffix(".props").read_bytes(), orig_props)


# ===========================================================================
# P5: Copy mode (--mode copy duplicates block files into <project>/blocks/)
# ===========================================================================


class TestCopyModeHelpers(unittest.TestCase):
    """Direct unit tests for copy_block_to_project / remove_block_from_project."""

    def test_copy_exclude_skips_listed_dirs(self):
        # Cinder-Serial's cinderblock.xml has <copyExclude>samples</copyExclude>.
        with _SameDriveTempDir() as tmp:
            proj = tmp / "P"
            (proj / "vc2022").mkdir(parents=True)
            m = _manifest("Cinder-Serial")
            dst = cb.copy_block_to_project(m, proj)
            self.assertTrue(dst.is_dir())
            self.assertTrue((dst / "src" / "SerialDevice.cpp").is_file())
            # samples dir must NOT be copied.
            self.assertFalse((dst / "samples").exists(),
                             f"samples should be excluded but exists at {dst}")
            # tidy up
            self.assertTrue(cb.remove_block_from_project(proj, m.folder_name))

    def test_remove_returns_false_when_not_present(self):
        with _SameDriveTempDir() as tmp:
            proj = tmp / "P"
            (proj / "vc2022").mkdir(parents=True)
            self.assertFalse(cb.remove_block_from_project(proj, "Whatever"))

    def test_copy_overwrites_existing_destination(self):
        # Re-copy should produce a clean mirror, not layered on prior contents.
        with _SameDriveTempDir() as tmp:
            proj = tmp / "P"
            (proj / "vc2022").mkdir(parents=True)
            m = _manifest("OSC")
            cb.copy_block_to_project(m, proj)
            # Pollute the destination with a stray file.
            stray = cb._local_block_dir(proj, m.folder_name) / "stray.txt"
            stray.write_text("noise")
            self.assertTrue(stray.exists())

            # Re-copy; stray file should be gone.
            cb.copy_block_to_project(m, proj)
            self.assertFalse(stray.exists())


class TestCmdCopyMode(unittest.TestCase):
    """End-to-end CLI tests for --mode copy."""

    def test_copy_mode_copies_files_and_uses_local_macro(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "CopyMode")
            project = vcx.parent.parent
            self.assertEqual(
                _run_cli("add", "OSC", "--project", str(project), "--mode", "copy"),
                0,
            )
            # Block files are now under <project>/blocks/OSC/.
            self.assertTrue((project / "blocks" / "OSC" / "src" / "cinder" / "osc" /
                             "Osc.cpp").is_file())
            self.assertTrue((project / "blocks" / "OSC" / "src" / "cinder" / "osc" /
                             "Osc.h").is_file())

            text = vcx.read_text(encoding="utf-8")
            # Macro value uses the local macro, not the shared tree.
            self.assertIn(
                r"<CinderBlocksIncludePaths>$(CinderBlocksLocalDir)\OSC\src</CinderBlocksIncludePaths>",
                text,
            )
            # ClCompile/ClInclude entries point at the local copy.
            self.assertIn(
                r'<ClCompile Include="$(CinderBlocksLocalDir)\OSC\src\cinder\osc\Osc.cpp" />',
                text,
            )
            self.assertIn(
                r'<ClInclude Include="$(CinderBlocksLocalDir)\OSC\src\cinder\osc\Osc.h" />',
                text,
            )

            # .props file defines CinderBlocksLocalDir.
            props_text = vcx.with_suffix(".props").read_text(encoding="utf-8")
            self.assertIn("<CinderBlocksLocalDir>", props_text)

            # Sidecar records the mode.
            sidecar = cb.load_sidecar(project)
            self.assertEqual(sidecar["blocks"]["OSC"]["mode"], "copy")

    def test_copy_mode_copy_exclude_honored(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "CopyExclude")
            project = vcx.parent.parent
            self.assertEqual(
                _run_cli("--blocks-dir", str(BLOCKS_DIR),
                         "add", "Cinder-Serial", "--project", str(project), "--mode", "copy"),
                0,
            )
            block_dir = project / "blocks" / "Cinder-Serial"
            self.assertTrue(block_dir.is_dir())
            self.assertFalse((block_dir / "samples").exists())

    def test_remove_copy_mode_deletes_local_dir(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "RemoveCopy")
            project = vcx.parent.parent
            _run_cli("add", "OSC", "--project", str(project), "--mode", "copy")
            self.assertTrue((project / "blocks" / "OSC").exists())

            self.assertEqual(
                _run_cli("remove", "OSC", "--project", str(project)),
                0,
            )
            self.assertFalse((project / "blocks" / "OSC").exists())
            # The empty blocks/ parent dir should also be cleaned up.
            self.assertFalse((project / "blocks").exists())

    def test_copy_mode_with_dependency_copies_dep_too(self):
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "CopyDep")
            project = vcx.parent.parent
            self.assertEqual(
                _run_cli("add", "TUIO", "--project", str(project), "--mode", "copy"),
                0,
            )
            # Both TUIO and its auto-pulled OSC dep should be copied locally.
            self.assertTrue((project / "blocks" / "TUIO").exists())
            self.assertTrue((project / "blocks" / "OSC").exists())

            text = vcx.read_text(encoding="utf-8")
            # Both blocks emit $(CinderBlocksLocalDir) paths.
            self.assertIn(r"$(CinderBlocksLocalDir)\TUIO", text)
            self.assertIn(r"$(CinderBlocksLocalDir)\OSC", text)
            # No reference-mode paths leak through.
            self.assertNotIn(r"$(CinderBlocksDir)\TUIO", text)
            self.assertNotIn(r"$(CinderBlocksDir)\OSC", text)

    def test_mixed_modes_in_same_project(self):
        # One block in reference mode, another in copy mode. Watchdog is
        # only present in CinderHead\blocks, so pass --blocks-dir explicitly.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "Mixed")
            project = vcx.parent.parent
            _run_cli("add", "OSC", "--project", str(project), "--mode", "reference")
            _run_cli("--blocks-dir", str(BLOCKS_DIR),
                     "add", "Watchdog", "--project", str(project), "--mode", "copy")

            # Only Watchdog is locally copied.
            self.assertFalse((project / "blocks" / "OSC").exists())
            self.assertTrue((project / "blocks" / "Watchdog").exists())

            text = vcx.read_text(encoding="utf-8")
            # IncludePaths mixes both anchors.
            self.assertIn(r"$(CinderBlocksDir)\OSC\src", text)
            self.assertIn(r"$(CinderBlocksLocalDir)\Watchdog\include", text)

            # Removing the copy-mode block leaves the reference-mode one intact.
            _run_cli("--blocks-dir", str(BLOCKS_DIR),
                     "remove", "Watchdog", "--project", str(project))
            self.assertFalse((project / "blocks" / "Watchdog").exists())
            sidecar = cb.load_sidecar(project)
            self.assertIn("OSC", sidecar["blocks"])
            self.assertNotIn("Watchdog", sidecar["blocks"])

    def test_update_resyncs_copy_mode_blocks(self):
        # `update` re-copies so files removed upstream disappear locally.
        with _SameDriveTempDir() as tmp:
            vcx = _generate_project(tmp, "UpdateResync")
            project = vcx.parent.parent
            _run_cli("add", "OSC", "--project", str(project), "--mode", "copy")

            # Pollute the local copy.
            stray = project / "blocks" / "OSC" / "stray.txt"
            stray.write_text("noise")
            self.assertTrue(stray.exists())

            # Update should re-sync and remove the stray file.
            self.assertEqual(_run_cli("update", "--project", str(project)), 0)
            self.assertFalse(stray.exists())
