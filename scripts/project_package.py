#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from zi2zi_webui.project_packages import (
    export_project_package,
    import_project_package,
    inspect_project_package,
)
from zi2zi_webui.storage import Storage


def emit_progress(current: int, total: int, message: str) -> None:
    print(
        "WEBUI_PROGRESS "
        + json.dumps(
            {"current": current, "total": max(total, 1), "message": message},
            ensure_ascii=False,
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Export or import a zi2zi-JiT project package")
    commands = root.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--data-dir", required=True)
    export.add_argument("--project-id", required=True)
    export.add_argument("--output", required=True)
    export.add_argument(
        "--mode",
        choices=["lightweight", "full"],
        default="lightweight",
    )
    export.add_argument("--include-shared-models", action="store_true")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--package", required=True)

    import_command = commands.add_parser("import")
    import_command.add_argument("--data-dir", required=True)
    import_command.add_argument("--package", required=True)
    import_command.add_argument("--delete-package-after", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "inspect":
        result = inspect_project_package(args.package, progress=emit_progress)
    elif args.command == "export":
        result = export_project_package(
            Storage(args.data_dir),
            args.project_id,
            args.output,
            mode=args.mode,
            include_shared_models=args.include_shared_models,
            progress=emit_progress,
        )
    else:
        try:
            result = import_project_package(
                Storage(args.data_dir),
                args.package,
                progress=emit_progress,
            )
        finally:
            if args.delete_package_after:
                Path(args.package).unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
