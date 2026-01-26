import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.unreal.remote_control import RemoteControlClient, RemoteControlError


def parse_json_arg(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unreal Remote Control CLI")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument("--timeout", type=float, default=5.0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="List Remote Control API routes")
    subparsers.add_parser("actors", help="List actors in current level")

    describe = subparsers.add_parser("describe", help="Describe a UObject")
    describe.add_argument("--object-path", required=True)

    get_prop = subparsers.add_parser("get", help="Read property values")
    get_prop.add_argument("--object-path", required=True)
    get_prop.add_argument("--property", required=False)

    set_prop = subparsers.add_parser("set", help="Write a property value")
    set_prop.add_argument("--object-path", required=True)
    set_prop.add_argument("--property", required=True)
    set_prop.add_argument("--value", required=True, type=parse_json_arg)
    set_prop.add_argument("--transaction", action="store_true")

    call_fn = subparsers.add_parser("call", help="Call a BlueprintCallable function")
    call_fn.add_argument("--object-path", required=True)
    call_fn.add_argument("--function", required=True)
    call_fn.add_argument("--params", default="{}", type=parse_json_arg)
    call_fn.add_argument("--transaction", action="store_true")

    search = subparsers.add_parser("search", help="Search assets")
    search.add_argument("--query", default="")
    search.add_argument("--package-paths", default="[]", type=parse_json_arg)
    search.add_argument("--class-names", default="[]", type=parse_json_arg)
    search.add_argument("--recursive-paths", action="store_true")
    search.add_argument("--recursive-classes", action="store_true")

    maps = subparsers.add_parser("maps", help="List level/map assets")
    maps.add_argument("--package-paths", default='["/Game"]', type=parse_json_arg)
    maps.add_argument("--recursive-paths", action="store_true")

    open_map = subparsers.add_parser("open-map", help="Open a level/map asset")
    open_map.add_argument("--asset-path", required=True)

    prune_maps = subparsers.add_parser(
        "prune-maps", help="Delete all maps except one"
    )
    prune_maps.add_argument("--keep", required=True, help="Asset path to keep")
    prune_maps.add_argument("--package-paths", default='["/Game"]', type=parse_json_arg)
    prune_maps.add_argument("--recursive-paths", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    client = RemoteControlClient(host=args.host, port=args.port, timeout=args.timeout)

    try:
        if args.command == "info":
            print_json(client.info())
        elif args.command == "actors":
            print_json(client.get_all_level_actors())
        elif args.command == "describe":
            print_json(client.describe_object(args.object_path))
        elif args.command == "get":
            print_json(client.get_properties(args.object_path, args.property))
        elif args.command == "set":
            print_json(
                client.set_property(
                    args.object_path, args.property, args.value, args.transaction
                )
            )
        elif args.command == "call":
            print_json(
                client.call_function(
                    args.object_path,
                    args.function,
                    args.params,
                    generate_transaction=args.transaction,
                )
            )
        elif args.command == "search":
            print_json(
                client.search_assets(
                    query=args.query,
                    package_paths=args.package_paths,
                    class_names=args.class_names,
                    recursive_paths=args.recursive_paths,
                    recursive_classes=args.recursive_classes,
                )
            )
        elif args.command == "maps":
            print_json(
                client.list_maps(
                    package_paths=args.package_paths,
                    recursive_paths=args.recursive_paths,
                )
            )
        elif args.command == "open-map":
            print_json(client.load_level(args.asset_path))
        elif args.command == "prune-maps":
            maps_payload = client.list_maps(
                package_paths=args.package_paths,
                recursive_paths=args.recursive_paths,
            )
            if isinstance(maps_payload, dict):
                assets = maps_payload.get("Maps", maps_payload.get("Assets", []))
            else:
                assets = maps_payload
            keep_path = args.keep
            if assets and isinstance(assets[0], str):
                keep_found = keep_path in assets
            else:
                keep_found = any(asset.get("Path") == keep_path for asset in assets)
            if not keep_found:
                raise SystemExit(f"Keep map not found in search results: {keep_path}")

            deleted = []
            for asset in assets:
                path = asset if isinstance(asset, str) else asset.get("Path")
                if not path or path == keep_path:
                    continue
                client.delete_asset(path)
                deleted.append(path)
            print_json({"kept": keep_path, "deleted": deleted})
    except RemoteControlError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
