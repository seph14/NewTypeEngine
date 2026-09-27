#!/usr/bin/env python3
"""
Material Callable Generator for NewTypeEngine

Generates hot-reloadable material callable DLL projects with pre-configured
Visual Studio project files. Each project contains SurfaceResolveFn callables
that modify SurfaceData after texture sampling.

Unlike compute shader projects (shader_generator.py), material callable DLLs:
  - Export registerMaterialCallables/unregisterMaterialCallables (not create/destroy)
  - Use RT_RUNTIME (not RT_RUNTIME_DLL)
  - Register callables via host-provided function pointers (not ShaderManager)
  - Are loaded by CallableDLLLoader

Usage:
    python material_generator.py MyMaterials
    python material_generator.py MyMaterials --callables checkerboard,iridescent
    python material_generator.py MyMaterials --callables checkerboard,iridescent,animated_roughness --count 5
    python material_generator.py --list
"""

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    print("Error: Jinja2 is required. Install with: pip install jinja2")
    sys.exit(1)


def discover_solution(root_dir: Path) -> Tuple[Path, str]:
    """Find the .sln file in vc2022/ and parse the main project name.

    The "main" project is the one whose .vcxproj lives directly in vc2022/,
    not under runtime_shaders/ or other subdirectories.

    Returns (sln_path, project_name).

    Raises FileNotFoundError if no .sln found.
    Raises RuntimeError if no main project can be identified.
    """
    vc2022 = root_dir / "vc2022"
    sln_files = list(vc2022.glob("*.sln"))
    if not sln_files:
        raise FileNotFoundError(f"No .sln file found in {vc2022}")

    sln_path = sln_files[0]

    # Parse Project() lines to find the main project
    # Format: Project("{GUID}") = "ProjectName", "relative\path.vcxproj", "{GUID}"
    project_pattern = re.compile(
        r'Project\("\{[^}]+\}"\)\s*=\s*"([^"]+)"\s*,\s*"([^"]+\.vcxproj)"'
    )

    for line in sln_path.read_text(encoding="utf-8-sig").splitlines():
        m = project_pattern.search(line)
        if not m:
            continue
        proj_name, proj_path = m.group(1), m.group(2)
        # Normalize to forward slashes
        proj_path = proj_path.replace("\\", "/")
        # Main project: .vcxproj directly in vc2022/, no subdirectories
        if not proj_path.startswith("..") and "/" not in proj_path:
            return sln_path, proj_name

    raise RuntimeError(
        f"Could not identify main project in {sln_path}. "
        f"Expected a .vcxproj directly in vc2022/ (not in a subdirectory)."
    )


def discover_namespace(root_dir: Path) -> Tuple[str, str]:
    """Discover the engine namespace and include path.

    Searches for the include/ directory first in the project root, then
    falls back to the sibling NewTypeEngine directory (for spawned projects
    at D:/Projects/OtherProject referencing D:/Projects/NewTypeEngine).

    Returns (namespace, engine_include_rel) where:
      - namespace: the engine namespace (e.g., "newtype")
      - engine_include_rel: relative path from runtime_shaders/ProjectName/
        to the engine's include/ directory (e.g., "..\\..\\include" or
        "..\\..\\..\\NewTypeEngine\\include")

    Raises FileNotFoundError if no suitable include/ found.
    Raises RuntimeError if namespace cannot be identified.
    """
    # Candidate paths: (include_dir, relative_from_runtime_shader_project)
    candidates = [
        (root_dir / "include", "..\\..\\include"),
        (root_dir.parent / "NewTypeEngine" / "include", "..\\..\\..\\NewTypeEngine\\include"),
    ]

    for include_dir, rel_path in candidates:
        if not include_dir.is_dir():
            continue

        for child in sorted(include_dir.iterdir()):
            if not child.is_dir():
                continue
            has_subdirs = any(d.is_dir() for d in child.iterdir())
            if has_subdirs:
                return child.name, rel_path

    tried = [str(c[0]) for c in candidates]
    raise FileNotFoundError(
        f"Could not find engine include/ directory with namespace subdirs. "
        f"Searched: {', '.join(tried)}"
    )


def inject_into_sln(sln_path: Path, project_name: str, vcxproj_rel: str, project_guid: str) -> None:
    """Add or update a project in the Visual Studio solution file.

    If a project with the same name already exists in the sln, the old entry
    (and its config platform entries) are removed before inserting the new one.
    """
    text = sln_path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    project_type_guid = "{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}"
    guid_braced = "{" + project_guid + "}"

    # --- Remove existing entries with same project name (handles --force re-gen with new GUID) ---
    project_pattern = re.compile(
        r'Project\("\{[^}]+\}"\)\s*=\s*"([^"]+)"\s*,\s*"([^"]+\.vcxproj)"\s*,\s*"\{([^}]+)\}"'
    )
    old_guids = []
    lines_to_remove = set()  # indices to remove
    for i, line in enumerate(lines):
        m = project_pattern.search(line)
        if m and m.group(1) == project_name:
            old_guids.append(m.group(3))
            lines_to_remove.add(i)
            # Also mark the EndProject on the next line
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip() == "EndProject":
                    lines_to_remove.add(j)
                    break

    if old_guids:
        # Remove old Project/EndProject lines
        lines = [l for i, l in enumerate(lines) if i not in lines_to_remove]

        # Remove old config platform entries (lines containing any old GUID)
        for old_guid in old_guids:
            old_guid_braced = "{" + old_guid + "}"
            lines = [l for l in lines if old_guid_braced not in l]

    # --- Insert Project() block before "Global" ---
    global_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "Global":
            global_idx = i
            break

    if global_idx is None:
        raise RuntimeError(f"No 'Global' section found in {sln_path}")

    lines.insert(global_idx, "EndProject")
    lines.insert(global_idx, f'Project("{project_type_guid}") = "{project_name}", "{vcxproj_rel}", "{guid_braced}"')

    # --- Insert ProjectConfigurationPlatforms entries ---
    config_entries = [
        f"\t\t{guid_braced}.Debug_Runtime|x64.ActiveCfg = Debug_Runtime|x64",
        f"\t\t{guid_braced}.Debug_Runtime|x64.Build.0 = Debug_Runtime|x64",
        f"\t\t{guid_braced}.Debug|x64.ActiveCfg = Debug|x64",
        f"\t\t{guid_braced}.Debug|x64.Build.0 = Debug|x64",
        f"\t\t{guid_braced}.Release|x64.ActiveCfg = Release|x64",
        f"\t\t{guid_braced}.Release|x64.Build.0 = Release|x64",
    ]

    # Find the end of the ProjectConfigurationPlatforms section
    in_proj_configs = False
    insert_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("GlobalSection(ProjectConfigurationPlatforms)"):
            in_proj_configs = True
        elif in_proj_configs and stripped == "EndGlobalSection":
            insert_idx = i
            break

    if insert_idx is None:
        raise RuntimeError(f"No ProjectConfigurationPlatforms section found in {sln_path}")

    for j, entry in enumerate(config_entries):
        lines.insert(insert_idx + j, entry)

    sln_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    action = "Updated" if old_guids else "Added"
    print(f"  {action} {project_name} in {sln_path.name}")


class MaterialGenerator:
    """Generator for hot-reloadable material callable DLL projects."""

    def __init__(self, root_dir: Path, project_name: str, namespace: str, engine_include_rel: str):
        self.root_dir = root_dir
        self.project_name = project_name
        self.namespace = namespace
        self.engine_include_rel = engine_include_rel
        self.template_dir = root_dir / "tools" / "shader_generator" / "templates"
        self.runtime_shaders_dir = root_dir / "runtime_shaders"
        self.registry_file = root_dir / "tools" / "shader_generator" / "materials.json"
        self.solution_file = root_dir / "vc2022" / f"{project_name}.sln"

        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True
        )

    def generate_guid(self) -> str:
        return str(uuid.uuid4()).upper().replace("-", "")

    def parse_callable_names(self, names_str: str, count: int) -> List[Dict[str, str]]:
        """Parse comma-separated callable names into template-ready dicts.

        When names are explicitly provided via --callables, all are used
        regardless of --count. When no names given, --count placeholder stubs
        are generated.
        """
        if names_str:
            names = [n.strip() for n in names_str.split(",") if n.strip()]
        else:
            names = [f"callable_{i+1}" for i in range(count)]

        return [
            {"name": name, "description": "Custom material effect"}
            for name in names
        ]

    def to_export_macro(self, name: str) -> str:
        """Convert a project name to an export macro.

        Handles PascalCase, snake_case, and CamelCase inputs:
          CustomMaterials -> CUSTOM_MATERIALS
          MyCoolEffect    -> MY_COOL_EFFECT
          test_project    -> TEST_PROJECT
        """
        import re
        # Insert underscore before uppercase letters that follow lowercase or before
        # an uppercase letter followed by a lowercase (for CamelCase boundaries)
        s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.upper()

    def load_registry(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"materials": []}

    def save_registry(self, registry: Dict[str, Any]) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def register_material(self, shader_name: str, export_macro: str, callable_names: List[str]) -> None:
        registry = self.load_registry()

        entry = {
            "name": shader_name,
            "export_macro": export_macro,
            "directory": f"runtime_shaders/{shader_name}",
            "callables": callable_names,
        }

        registry["materials"] = [m for m in registry["materials"] if m["name"] != shader_name]
        registry["materials"].append(entry)
        self.save_registry(registry)

    def generate(
        self,
        name: str,
        callables: Optional[List[Dict[str, str]]] = None,
        description: Optional[str] = None,
        force: bool = False
    ) -> bool:
        # Normalize name — used as-is for directory and file names
        shader_name = name.strip().replace(" ", "_")

        export_macro = self.to_export_macro(shader_name)
        output_dir = self.runtime_shaders_dir / shader_name

        if output_dir.exists() and not force:
            print(f"Error: Directory '{output_dir}' already exists.")
            print(f"Use --force to overwrite existing files.")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        # Default to 1 callable if none specified
        if callables is None:
            callables = [{"name": "custom_effect", "description": "Custom material effect"}]

        context = {
            "shader_name": shader_name,
            "export_macro": export_macro,
            "project_guid": self.generate_guid(),
            "project_name": self.project_name,
            "namespace": self.namespace,
            "engine_include_rel": self.engine_include_rel,
            "callables": callables,
            "description": description or f"{shader_name} custom material callables with hot-reload",
            # PCH payload: the heavy template headers that dominate the TU
            # compile. Mirrors runtime_shaders/CustomMaterialShader/pch.h.
            "pch_includes": [
                f'"{shader_name}API.h"',
                f'"{self.namespace}/render/Shading.h"',
                "<luisa/dsl/sugar.h>",
                "<cstdio>",
            ],
        }

        templates = [
            ("material.vcxproj.j2", f"{shader_name}.vcxproj"),
            ("material.h.j2", f"{shader_name}.h"),
            ("materialAPI.h.j2", f"{shader_name}API.h"),
            ("material.cpp.j2", f"{shader_name}.cpp"),
            ("pch.h.j2", "pch.h"),
            ("pch.cpp.j2", "pch.cpp"),
        ]

        for template_name, output_name in templates:
            template = self.env.get_template(template_name)
            rendered = template.render(**context)

            output_path = output_dir / output_name
            with open(output_path, "w", newline="\n") as f:
                f.write(rendered)
            print(f"Created: {output_path}")

        # Register
        callable_names = [c["name"] for c in callables]
        self.register_material(shader_name, export_macro, callable_names)

        # Inject into solution
        project_guid = context["project_guid"]
        vcxproj_rel = f"..\\runtime_shaders\\{shader_name}\\{shader_name}.vcxproj"
        try:
            inject_into_sln(self.solution_file, shader_name, vcxproj_rel, project_guid)
        except Exception as e:
            print(f"  Warning: Could not update solution file: {e}")
            print(f"  Add manually: {output_dir / f'{shader_name}.vcxproj'}")

        # Print instructions
        print(f"\n{'='*60}")
        print(f"Material callable project '{shader_name}' created!")
        print(f"{'='*60}")
        print(f"\nCallables: {', '.join(callable_names)}")
        print(f"\nFiles created in: {output_dir}")
        print(f"\nTo use in code (static mode):")
        print(f'  #include "{shader_name}.h"')
        print(f"  // Callables are registered automatically by the DLL loader")
        print(f"\nTo use in code (runtime mode):")
        print(f"  // CallableDLLLoader handles hot-reload automatically")
        print(f"  // Assign custom material types to shapes via:")
        print(f'  // mat.setCustomCallable("callable_name")')
        print(f"\nEdit {output_dir / f'{shader_name}.cpp'} to implement your effects.")

        return True

    def list_materials(self) -> None:
        registry = self.load_registry()

        if not registry["materials"]:
            print("No material callable projects registered.")
            return

        print("Registered material callable projects:")
        print("-" * 60)
        for mat in registry["materials"]:
            print(f"  Name: {mat['name']}")
            print(f"  Export Macro: {mat['export_macro']}")
            print(f"  Directory: {mat['directory']}")
            print(f"  Callables: {', '.join(mat.get('callables', []))}")
            print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hot-reloadable material callable DLL projects for NewTypeEngine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: create CustomMaterialShader with one callable stub
  python material_generator.py

  # Default name, multiple callables
  python material_generator.py --callables checkerboard,iridescent,wave

  # Default name, 5 placeholder stubs
  python material_generator.py --count 5

  # Custom project name
  python material_generator.py --name MyMaterials --callables glow,pulse

  # List all registered material projects
  python material_generator.py --list

  # Overwrite existing project
  python material_generator.py --force
        """
    )

    parser.add_argument(
        "--name",
        default="CustomMaterialShader",
        help="Project name (default: CustomMaterialShader)"
    )

    parser.add_argument(
        "--callables",
        help="Comma-separated callable names (e.g., 'checkerboard,iridescent,wave')"
    )

    parser.add_argument(
        "--count", "-n",
        type=int,
        default=1,
        help="Number of callable stubs to generate (default: 1, ignored if --callables has more)"
    )

    parser.add_argument(
        "--description",
        help="Description of the material project"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing files"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered material callable projects"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    root_dir = Path(__file__).resolve().parents[2]

    try:
        sln_path, project_name = discover_solution(root_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run this script from the project root directory.")
        return 1
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    try:
        namespace, engine_include_rel = discover_namespace(root_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    generator = MaterialGenerator(root_dir, project_name, namespace, engine_include_rel)

    if args.list:
        generator.list_materials()
        return 0

    callables = generator.parse_callable_names(args.callables or "", args.count)

    success = generator.generate(
        name=args.name,
        callables=callables,
        description=args.description,
        force=args.force
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
