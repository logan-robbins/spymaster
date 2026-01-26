import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class RemoteControlError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteControlConfig:
    host: str = "127.0.0.1"
    port: int = 30010
    timeout: float = 5.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RemoteControlClient:
    def __init__(
        self, host: str = "127.0.0.1", port: int = 30010, timeout: float = 5.0
    ) -> None:
        self.config = RemoteControlConfig(host=host, port=port, timeout=timeout)

    def info(self) -> Any:
        return self._request("GET", "/remote/info")

    def call_function(
        self,
        object_path: str,
        function_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        generate_transaction: bool = False,
    ) -> Any:
        payload = {
            "objectPath": object_path,
            "functionName": function_name,
            "parameters": parameters or {},
            "generateTransaction": generate_transaction,
        }
        return self._request("PUT", "/remote/object/call", payload)

    def describe_object(self, object_path: str) -> Any:
        payload = {"objectPath": object_path}
        return self._request("PUT", "/remote/object/describe", payload)

    def get_properties(self, object_path: str, property_name: Optional[str] = None) -> Any:
        payload: Dict[str, Any] = {
            "objectPath": object_path,
            "access": "READ_ACCESS",
        }
        if property_name:
            payload["propertyName"] = property_name
            payload["property"] = property_name
        return self._request("PUT", "/remote/object/property", payload)

    def set_property(
        self,
        object_path: str,
        property_name: str,
        value: Any,
        transaction: bool = False,
    ) -> Any:
        payload = {
            "objectPath": object_path,
            "access": "WRITE_TRANSACTION_ACCESS" if transaction else "WRITE_ACCESS",
            "propertyName": property_name,
            "propertyValue": {property_name: value},
        }
        return self._request("PUT", "/remote/object/property", payload)

    def set_properties(
        self,
        object_path: str,
        values: Dict[str, Any],
        transaction: bool = False,
    ) -> Any:
        if not values:
            raise RemoteControlError("values must be a non-empty dict")
        payload = {
            "objectPath": object_path,
            "access": "WRITE_TRANSACTION_ACCESS" if transaction else "WRITE_ACCESS",
            "propertyValue": values,
        }
        return self._request("PUT", "/remote/object/property", payload)

    def search_assets(
        self,
        query: str = "",
        package_paths: Optional[list[str]] = None,
        class_names: Optional[list[str]] = None,
        recursive_paths: bool = False,
        recursive_classes: bool = False,
    ) -> Any:
        payload = {
            "Query": query,
            "Filter": {
                "PackageNames": [],
                "ClassNames": class_names or [],
                "PackagePaths": package_paths or [],
                "RecursiveClassesExclusionSet": [],
                "RecursivePaths": recursive_paths,
                "RecursiveClasses": recursive_classes,
            },
        }
        return self._request("PUT", "/remote/search/assets", payload)

    def get_all_level_actors(self) -> Any:
        return self.call_function(
            "/Script/UnrealEd.Default__EditorActorSubsystem",
            "GetAllLevelActors",
        )

    def list_maps(
        self,
        package_paths: Optional[list[str]] = None,
        recursive_paths: bool = True,
    ) -> Any:
        map_paths: set[str] = set()
        for root in package_paths or ["/Game"]:
            assets = self.call_function(
                "/Script/EditorScriptingUtilities.Default__EditorAssetLibrary",
                "ListAssets",
                {
                    "DirectoryPath": root,
                    "bRecursive": recursive_paths,
                    "bIncludeFolder": False,
                },
            )
            asset_paths = assets.get("ReturnValue", assets) if isinstance(assets, dict) else assets
            for path in asset_paths or []:
                if ":PersistentLevel" in path:
                    map_paths.add(path.split(":")[0])

        return {
            "Maps": [
                {"Path": path, "Name": path.split("/")[-1].split(".")[0]}
                for path in sorted(map_paths)
            ]
        }

    def load_level(self, asset_path: str) -> Any:
        return self.call_function(
            "/Script/LevelEditor.Default__LevelEditorSubsystem",
            "LoadLevel",
            {"AssetPath": asset_path},
        )

    def delete_asset(self, asset_path: str) -> Any:
        return self.call_function(
            "/Script/EditorScriptingUtilities.Default__EditorAssetLibrary",
            "DeleteAsset",
            {"AssetPathToDelete": asset_path},
        )

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.config.base_url}{path}"
        data = None
        headers = {"Content-Type": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8") if exc.fp else ""
            raise RemoteControlError(
                f"Remote Control HTTP {exc.code} for {path}: {raw}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteControlError(
                f"Remote Control connection failed: {exc}"
            ) from exc
