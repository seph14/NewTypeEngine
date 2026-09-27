#!/usr/bin/env python3
"""
Shader Project Generator for NewTypeEngine

Generates hot-reloadable shader projects with pre-configured Visual Studio
project files, DLL export interface, and template shader code.

Usage:
    python shader_generator.py MyShader
    python shader_generator.py MyShader --params "ImageFloat output, ImageUInt seed"
    python shader_generator.py --list
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

    project_pattern = re.compile(
        r'Project\("\{[^}]+\}"\)\s*=\s*"([^"]+)"\s*,\s*"([^"]+\.vcxproj)"'
    )

    for line in sln_path.read_text(encoding="utf-8-sig").splitlines():
        m = project_pattern.search(line)
        if not m:
            continue
        proj_name, proj_path = m.group(1), m.group(2)
        proj_path = proj_path.replace("\\", "/")
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
    lines_to_remove = set()
    for i, line in enumerate(lines):
        m = project_pattern.search(line)
        if m and m.group(1) == project_name:
            old_guids.append(m.group(3))
            lines_to_remove.add(i)
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip() == "EndProject":
                    lines_to_remove.add(j)
                    break

    if old_guids:
        lines = [l for i, l in enumerate(lines) if i not in lines_to_remove]
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


class ShaderGenerator:
    """Generator for hot-reloadable shader projects."""

    def __init__(self, root_dir: Path, project_name: str, engine_include_rel: str):
        self.root_dir = root_dir
        self.project_name = project_name
        self.engine_include_rel = engine_include_rel
        self.template_dir = root_dir / "tools" / "shader_generator" / "templates"
        self.runtime_shaders_dir = root_dir / "runtime_shaders"
        self.registry_file = root_dir / "tools" / "shader_generator" / "shaders.json"
        self.solution_file = root_dir / "vc2022" / f"{project_name}.sln"

        # Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Custom filters
        self.env.filters["lower"] = lambda s: s.lower()

    def generate_guid(self) -> str:
        """Generate a unique GUID for the Visual Studio project."""
        return str(uuid.uuid4()).upper().replace("-", "")

    def parse_kernel_params(self, params_str: str) -> List[str]:
        """Parse kernel parameter string into individual parameters."""
        if not params_str:
            # Default parameters for a simple compute shader
            return [
                "luisa::compute::Image<float>",   # output
                "luisa::compute::Image<uint>",    # seed
                "luisa::compute::Accel",          # accel
                "uint",                           # frame
            ]

        # Split by comma and clean up
        params = [p.strip() for p in params_str.split(",")]
        return [p for p in params if p]

    def generate_forward_declarations(self, params: List[str]) -> List[str]:
        """Generate forward declarations for custom types in parameters."""
        declarations = []
        for param in params:
            # Extract namespace::type from parameters
            if "::" in param and not param.startswith("luisa::"):
                # This is a custom type, add forward declaration
                type_name = param.split()[-1].rstrip(">")
                namespace = "::".join(type_name.split("::")[:-1]) if "::" in type_name else ""
                type_only = type_name.split("::")[-1]

                if namespace and "Var<" not in param:
                    declarations.append(f"namespace {namespace} {{")
                    declarations.append(f"    struct {type_only};")
                    declarations.append(f"}}")
        return declarations

    def load_registry(self) -> Dict[str, Any]:
        """Load the shader registry JSON file."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"shaders": []}

    def save_registry(self, registry: Dict[str, Any]) -> None:
        """Save the shader registry JSON file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def register_shader(self, base_name: str, class_name: str, export_macro: str, shader_name: str) -> None:
        """Add a new shader to the registry."""
        registry = self.load_registry()

        # Check for duplicates
        for shader in registry["shaders"]:
            if shader["name"] == base_name:
                print(f"Warning: Shader '{base_name}' already registered. Overwriting.")

        # Add or update the shader entry
        # base_name is the name to use in DLLHotReload (e.g., "PathTracer")
        # shader_name is the directory name (e.g., "PathTracerShader")
        entry = {
            "name": base_name,
            "class_name": class_name,
            "export_macro": export_macro,
            "directory": f"runtime_shaders/{shader_name}",
        }

        # Remove existing entry with same name
        registry["shaders"] = [s for s in registry["shaders"] if s["name"] != base_name]
        registry["shaders"].append(entry)

        self.save_registry(registry)

    def generate_shader(
        self,
        name: str,
        params: Optional[List[str]] = None,
        description: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """Generate a new shader project."""
        # Normalize shader name
        input_name = name.strip().replace(" ", "_")

        # Determine class name (always append "Shader" if not already present)
        if input_name.endswith("Shader"):
            class_name = input_name
        else:
            class_name = input_name + "Shader"

        # Function suffix is the base name WITHOUT "Shader" suffix
        # e.g., "PathTracerShader" -> "PathTracer"
        if class_name.endswith("Shader"):
            function_suffix = class_name[:-6]  # Remove "Shader" (6 chars)
        else:
            function_suffix = class_name

        # Export macro is the uppercased function suffix
        # e.g., "PathTracer" -> "PATH_TRACER"
        export_macro = function_suffix.upper()

        # For file naming and directory, use the full class name
        shader_name = class_name

        # Set output directory
        output_dir = self.runtime_shaders_dir / shader_name

        # Check if directory exists
        if output_dir.exists() and not force:
            print(f"Error: Directory '{output_dir}' already exists.")
            print(f"Use --force to overwrite existing files.")
            return False

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse kernel parameters
        if params is None:
            params = self.parse_kernel_params("")

        # Generate forward declarations
        forward_declarations = self.generate_forward_declarations(params)

        # Template context
        context = {
            "shader_name": shader_name,
            "class_name": class_name,
            "export_macro": export_macro,
            "function_suffix": function_suffix,
            "project_guid": self.generate_guid(),
            "project_name": self.project_name,
            "engine_include_rel": self.engine_include_rel,
            "kernel_params": params,
            "forward_declarations": forward_declarations,
            "description": description,
            "additional_includes": [],
            "using_namespaces": [],
            # PCH payload: the heavy template headers that dominate the TU
            # compile. Mirrors runtime_shaders/CustomMaterialShader/pch.h.
            "pch_includes": [
                f'"{shader_name}.h"',
                "<luisa/dsl/sugar.h>",
            ],
        }

        # Render and write templates
        templates = [
            ("shader.vcxproj.j2", f"{shader_name}.vcxproj"),
            ("shader.h.j2", f"{shader_name}.h"),
            ("shaderAPI.h.j2", f"{shader_name}API.h"),
            ("shader.cpp.j2", f"{shader_name}.cpp"),
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

        # Register the shader (use function_suffix as the base name for ShaderManager)
        self.register_shader(function_suffix, class_name, export_macro, shader_name)

        # Inject into solution
        project_guid = context["project_guid"]
        vcxproj_rel = f"..\\runtime_shaders\\{shader_name}\\{shader_name}.vcxproj"
        try:
            inject_into_sln(self.solution_file, shader_name, vcxproj_rel, project_guid)
        except Exception as e:
            print(f"  Warning: Could not update solution file: {e}")
            print(f"  Add manually: {output_dir / f'{shader_name}.vcxproj'}")

        # Print usage instructions
        print(f"\n{'='*60}")
        print(f"Shader '{function_suffix}' created successfully!")
        print(f"{'='*60}")
        print(f"\nFiles created in: {output_dir}")
        print(f"\nTo use in code:")
        print(f'  #include "{shader_name}.h"')
        print(f"  // Load the shader (handles both static and DLL modes)")
        print(f"  {class_name} generator;")
        print(f"  ShaderManager::instance().loadShader(generator);")
        print(f"  ")
        print(f"  // Invoke the shader (specify template args matching your kernel)")
        print(f"  // Resolve once (member); dispatch has no string lookup")
        print(f"  ShaderHandle<2, /* Args... */> _{function_suffix};")
        print(f"  _{function_suffix}.assign(\"{function_suffix}\");")
        print(f"  stream << ShaderManager::instance().shader(_{function_suffix}, args...)")
        print(f"             .dispatch(width, height);")

        return True

    def list_shaders(self) -> None:
        """List all registered shaders."""
        registry = self.load_registry()

        if not registry["shaders"]:
            print("No shaders registered.")
            return

        print("Registered shaders:")
        print("-" * 60)
        for shader in registry["shaders"]:
            print(f"  Name: {shader['name']}")
            print(f"  Class: {shader['class_name']}")
            print(f"  Directory: {shader['directory']}")
            print()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate hot-reloadable shader projects for NewTypeEngine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new shader with default parameters
  python shader_generator.py MyShader

  # Create a shader with custom kernel parameters
  python shader_generator.py MyShader --params "ImageFloat output, ImageUInt seed, Accel accel"

  # List all registered shaders
  python shader_generator.py --list

  # Overwrite existing shader
  python shader_generator.py MyShader --force
        """
    )

    parser.add_argument(
        "shader_name",
        nargs="?",
        help="Name of the shader to generate"
    )

    parser.add_argument(
        "--params",
        help="Comma-separated kernel parameters (e.g., 'ImageFloat output, uint frame')"
    )

    parser.add_argument(
        "--description",
        help="Description of the shader"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing files"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered shaders"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Find root directory and discover solution
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
        _namespace, engine_include_rel = discover_namespace(root_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    generator = ShaderGenerator(root_dir, project_name, engine_include_rel)

    # Handle --list
    if args.list:
        generator.list_shaders()
        return 0

    # Require shader_name
    if not args.shader_name:
        parser.print_help()
        return 1

    # Parse kernel parameters
    params = None
    if args.params:
        params = generator.parse_kernel_params(args.params)

    # Generate the shader
    success = generator.generate_shader(
        name=args.shader_name,
        params=params,
        description=args.description,
        force=args.force
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
