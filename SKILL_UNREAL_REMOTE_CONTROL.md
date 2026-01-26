# Claude Skill: Unreal Engine 5 Development via Remote Control API

## Overview

This skill enables Claude to interact with Unreal Engine 5 (tested on 5.7) through the Remote Control API plugin. It covers C++ development, Blueprint automation, data streaming via UDP, and real-time visualization.

---

## 1. Remote Control API Setup

### Prerequisites
- Unreal Engine 5.x with Remote Control API plugin enabled
- Plugin: `Edit > Plugins > Remote Control API` (enable and restart)
- Default endpoint: `http://localhost:30010/remote`

### Python Client Implementation

This is the actual implementation we built (stdlib-only, no requests dependency):

```python
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
        """Get engine info and verify connection."""
        return self._request("GET", "/remote/info")

    def call_function(
        self,
        object_path: str,
        function_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        generate_transaction: bool = False,
    ) -> Any:
        """Call a UFUNCTION on an object."""
        payload = {
            "objectPath": object_path,
            "functionName": function_name,
            "parameters": parameters or {},
            "generateTransaction": generate_transaction,
        }
        return self._request("PUT", "/remote/object/call", payload)

    def describe_object(self, object_path: str) -> Any:
        """Get object metadata and available functions/properties."""
        payload = {"objectPath": object_path}
        return self._request("PUT", "/remote/object/describe", payload)

    def get_properties(self, object_path: str, property_name: Optional[str] = None) -> Any:
        """Get UPROPERTY value(s) from an object."""
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
        """Set a single UPROPERTY value."""
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
        """Set multiple UPROPERTY values at once."""
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
        """Search Content Browser for assets."""
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
        """Get all actors in the current level (Editor only)."""
        return self.call_function(
            "/Script/UnrealEd.Default__EditorActorSubsystem",
            "GetAllLevelActors",
        )

    def list_maps(
        self,
        package_paths: Optional[list[str]] = None,
        recursive_paths: bool = True,
    ) -> Any:
        """List all map assets in the project."""
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
        """Load a level/map in Editor."""
        return self.call_function(
            "/Script/LevelEditor.Default__LevelEditorSubsystem",
            "LoadLevel",
            {"AssetPath": asset_path},
        )

    def delete_asset(self, asset_path: str) -> Any:
        """Delete an asset (use with caution!)."""
        return self.call_function(
            "/Script/EditorScriptingUtilities.Default__EditorAssetLibrary",
            "DeleteAsset",
            {"AssetPathToDelete": asset_path},
        )

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        """Internal: Make HTTP request to Remote Control API."""
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
```

### CLI Pattern

```python
#!/usr/bin/env python
"""remote_control_cli.py - CLI for UE5 Remote Control"""
import argparse
from remote_control import RemoteControlClient

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    # info command
    subparsers.add_parser("info", help="Get engine info")
    
    # actors command
    subparsers.add_parser("actors", help="List all actors in level")
    
    # get command
    get_parser = subparsers.add_parser("get", help="Get property value")
    get_parser.add_argument("object_path")
    get_parser.add_argument("property_name")
    
    # set command
    set_parser = subparsers.add_parser("set", help="Set property value")
    set_parser.add_argument("object_path")
    set_parser.add_argument("property_name")
    set_parser.add_argument("value")
    
    args = parser.parse_args()
    client = RemoteControlClient()
    
    if args.command == "info":
        print(client.info())
    # ... etc

if __name__ == "__main__":
    main()
```

---

## 2. Object Path Conventions

### Level Actors
```
/Game/Maps/MyMap.MyMap:PersistentLevel.ActorName_0
```

### Blueprint Instances (in level)
```
/Game/Maps/MyMap.MyMap:PersistentLevel.BP_MyActor_C_0
```
Note: Blueprint class instances have `_C_N` suffix (C = compiled, N = instance number)

### Default Objects (for static functions)
```
/Script/Engine.Default__GameplayStatics
/Script/UnrealEd.Default__EditorLevelLibrary
/Script/Engine.Default__KismetSystemLibrary
```

### Asset References
```
/Game/Blueprints/BP_MyActor.BP_MyActor_C
/Game/Materials/M_MyMaterial.M_MyMaterial
/Game/Maps/MyMap.MyMap
```

---

## 3. C++ Development Patterns

### Actor with UDP Receiver

```cpp
// MwtUdpReceiver.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "MwtUdpReceiver.generated.h"

UCLASS()
class MYPROJECT_API AMwtUdpReceiver : public AActor {
  GENERATED_BODY()

public:
  AMwtUdpReceiver();

protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
  virtual void Tick(float DeltaTime) override;

  // === Configuration (exposed to Editor/Blueprints) ===
  
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Network")
  int32 Port = 7777;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
  bool bLogPackets = false;

  // === State (read-only in Blueprints) ===
  
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "State")
  int32 PacketsReceived = 0;

  // === Blueprint-callable functions ===
  
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void ResetState();

private:
  FSocket* ListenerSocket = nullptr;
  TSharedPtr<FInternetAddr> RemoteAddr;
  FCriticalSection DataMutex;
  
  void ProcessPacket(const TArray<uint8>& Data);
};
```

```cpp
// MwtUdpReceiver.cpp
#include "MwtUdpReceiver.h"
#include "Common/UdpSocketBuilder.h"

AMwtUdpReceiver::AMwtUdpReceiver() {
  PrimaryActorTick.bCanEverTick = true;
}

void AMwtUdpReceiver::BeginPlay() {
  Super::BeginPlay();
  
  // Create UDP socket
  FIPv4Endpoint Endpoint(FIPv4Address::Any, Port);
  ListenerSocket = FUdpSocketBuilder(TEXT("MwtSocket"))
    .AsNonBlocking()
    .AsReusable()
    .BoundToEndpoint(Endpoint)
    .WithReceiveBufferSize(1024 * 1024)
    .Build();
  
  if (!ListenerSocket) {
    UE_LOG(LogTemp, Error, TEXT("Failed to create UDP socket on port %d"), Port);
    return;
  }
  
  RemoteAddr = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
  UE_LOG(LogTemp, Log, TEXT("UDP Receiver listening on port %d"), Port);
}

void AMwtUdpReceiver::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (ListenerSocket) {
    ListenerSocket->Close();
    ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ListenerSocket);
    ListenerSocket = nullptr;
  }
  Super::EndPlay(EndPlayReason);
}

void AMwtUdpReceiver::Tick(float DeltaTime) {
  Super::Tick(DeltaTime);
  
  if (!ListenerSocket) return;
  
  TArray<uint8> RecvBuffer;
  RecvBuffer.SetNumUninitialized(65535);
  int32 BytesRead = 0;
  
  while (ListenerSocket->RecvFrom(RecvBuffer.GetData(), RecvBuffer.Num(), 
                                   BytesRead, *RemoteAddr)) {
    if (BytesRead > 0) {
      RecvBuffer.SetNum(BytesRead);
      ProcessPacket(RecvBuffer);
      PacketsReceived++;
    }
  }
}

void AMwtUdpReceiver::ProcessPacket(const TArray<uint8>& Data) {
  FScopeLock Lock(&DataMutex);
  // Parse packet here
  if (bLogPackets) {
    UE_LOG(LogTemp, Log, TEXT("Received %d bytes"), Data.Num());
  }
}

void AMwtUdpReceiver::ResetState() {
  FScopeLock Lock(&DataMutex);
  PacketsReceived = 0;
}
```

### Build.cs Module Dependencies

```csharp
// MyProject.Build.cs
using UnrealBuildTool;

public class MyProject : ModuleRules {
  public MyProject(ReadOnlyTargetRules Target) : base(Target) {
    PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

    PublicDependencyModuleNames.AddRange(new string[] { 
      "Core", 
      "CoreUObject", 
      "Engine", 
      "InputCore", 
      "EnhancedInput",
      // Network
      "Networking", 
      "Sockets",
      // Visualization
      "Niagara",
      "ProceduralMeshComponent"
    });

    PrivateDependencyModuleNames.AddRange(new string[] { });
  }
}
```

---

## 4. Common Errors and Fixes

### Error: "FObjectFinders can't be used outside of constructors"

**Exact Error Message**:
```
Fatal error: [File:./Runtime/CoreUObject/Private/UObject/UObjectGlobals.cpp] [Line: 5092] 
FObjectFinders can't be used outside of constructors to find /Engine/EngineMaterials/DefaultMaterial

0x0266a094 UnrealEditor-CoreUObject.dylib!ConstructorHelpers::CheckIfIsInConstructor(char16_t const*) [UnknownFile]) 
0x5a83de8c UnrealEditor-MarketWindTunnel.dylib!ConstructorHelpers::FObjectFinder<UMaterial>::FObjectFinder(char16_t const*, unsigned int)
0x5a83af94 UnrealEditor-MarketWindTunnel.dylib!UMwtHeatmapRenderer::CreateMaterials() [UnknownFile]) 
0x5a83aa38 UnrealEditor-MarketWindTunnel.dylib!UMwtHeatmapRenderer::BeginPlay() [UnknownFile])
```

**Cause**: Using `ConstructorHelpers::FObjectFinder<T>` in `BeginPlay()` or other runtime functions.

**Wrong**:
```cpp
void UMyComponent::BeginPlay() {
  Super::BeginPlay();
  // CRASH! FObjectFinder only works in constructors
  static ConstructorHelpers::FObjectFinder<UMaterial> MatFinder(TEXT("/Engine/..."));
}
```

**Correct**:
```cpp
// Option 1: Use in constructor (static lifetime)
UMyComponent::UMyComponent() {
  static ConstructorHelpers::FObjectFinder<UMaterial> MatFinder(TEXT("/Engine/..."));
  if (MatFinder.Succeeded()) {
    BaseMaterial = MatFinder.Object;
  }
}

// Option 2: Load at runtime with LoadObject
void UMyComponent::BeginPlay() {
  Super::BeginPlay();
  BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Engine/EngineMaterials/DefaultMaterial"));
}

// Option 3: Use soft references (best for assets)
UPROPERTY(EditDefaultsOnly)
TSoftObjectPtr<UMaterial> MaterialAsset;

void UMyComponent::BeginPlay() {
  if (MaterialAsset.IsValid()) {
    UMaterial* Mat = MaterialAsset.LoadSynchronous();
  }
}
```

### Error: Remote Control property not found (HTTP 400)

**Cause**: C++ code in workspace doesn't match compiled binary in UE project.

**Fix**: 
1. Verify which version is actually compiled: check `.cpp/.h` in UE project's `Source/` folder
2. Sync files: copy from UE project TO workspace (not the other way) if UE has newer code
3. Or copy from workspace TO UE project and force rebuild

### Error: Niagara "DataInterface was not found"

**Cause**: Niagara System lacks User Parameters that C++ code expects.

**Exact Error Messages** (repeated for each parameter):
```
LogNiagara: Warning: OverrideParameter(User.WallAsk) System(NS_MarketWindTunnel) DataInterface(NiagaraDataInterfaceArrayFloat) was not found
LogNiagara: Warning: OverrideParameter(User.WallBid) System(NS_MarketWindTunnel) DataInterface(NiagaraDataInterfaceArrayFloat) was not found
LogNiagara: Warning: OverrideParameter(User.GexImbalance) System(NS_MarketWindTunnel) DataInterface(NiagaraDataInterfaceArrayFloat) was not found
```

**C++ Code Causing This**:
```cpp
// This tries to push data to User Parameters that don't exist in the Niagara System
#include "NiagaraDataInterfaceArrayFunctionLibrary.h"

UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayFloat(
    NiagaraComp, FName("User.WallAsk"), WallAskArray);
```

**Why It Happens**: The Niagara System asset (`NS_MarketWindTunnel`) was created but:
- No User Parameters were added in the Niagara Editor
- Or User Parameters were added but with wrong types (need Float Array Data Interface)

**Fix Options**:
1. **Manual Niagara Configuration**: Open Niagara Editor → System Overview → User Parameters:
   - Add `User.WallAsk` (Float Array)
   - Add `User.WallBid` (Float Array)
   - Add `User.Vacuum` (Float Array)
   - Add `User.PhysicsSigned` (Float Array)
   - Add `User.GexAbs` (Float Array)
   - Add `User.GexImbalance` (Float Array)

2. **Alternative - Use Debug Draw** (faster for prototyping): See Section 6
   - No Niagara configuration needed
   - All visualization done via `DrawDebugSolidBox`
   - Works immediately after C++ compile

### Build Artifacts Causing Stale Code

**Symptoms**: Code changes not taking effect, old crashes repeating.

**Fix**: Delete ALL build artifacts:
```bash
cd "/path/to/UnrealProject"
rm -rf Binaries
rm -rf Intermediate
rm -rf DerivedDataCache
rm -rf Saved/ShaderDebugInfo
rm -rf Saved/StagedBuilds
# Then reopen UE Editor for full rebuild
```

---

## 5. UDP Protocol Design

### Packet Header Structure

```python
# Python packing
import struct

MAGIC = b'MWT1'  # 4-byte magic number
VERSION = 1

def pack_header(surface_id: int, payload_type: int, num_entries: int,
                window_end_ts_ns: int, spot_ref_price_int: int) -> bytes:
    """
    Header: 32 bytes
    - magic: 4 bytes (MWT1)
    - version: 1 byte
    - surface_id: 1 byte (0=snap, 1=wall, 2=vacuum, 3=physics, 4=gex)
    - payload_type: 1 byte
    - reserved: 1 byte
    - num_entries: 4 bytes (uint32)
    - window_end_ts_ns: 8 bytes (uint64)
    - spot_ref_price_int: 4 bytes (int32, price in ticks)
    - padding: 8 bytes
    """
    return struct.pack('<4sBBBBI Q i 8x',
        MAGIC, VERSION, surface_id, payload_type, 0,
        num_entries, window_end_ts_ns, spot_ref_price_int)
```

```cpp
// C++ unpacking
#pragma pack(push, 1)
struct FMwtHeader {
  uint8 Magic[4];      // "MWT1"
  uint8 Version;
  uint8 SurfaceId;     // 0=snap, 1=wall, 2=vacuum, 3=physics, 4=gex
  uint8 PayloadType;
  uint8 Reserved;
  uint32 NumEntries;
  uint64 WindowEndTsNs;
  int32 SpotRefPriceInt;  // Price in ticks
  uint8 Padding[8];
};
#pragma pack(pop)

bool ValidateHeader(const FMwtHeader& H) {
  return H.Magic[0] == 'M' && H.Magic[1] == 'W' && 
         H.Magic[2] == 'T' && H.Magic[3] == '1' &&
         H.Version == 1;
}
```

### Surface Payloads

```cpp
// Wall: sparse entries (tick offset + intensity)
#pragma pack(push, 1)
struct FWallEntry {
  int16 TickOffset;    // Relative to spot (-400 to +400)
  uint16 AskIntensity; // 0-65535 (log-scaled)
  uint16 BidIntensity;
  uint16 Reserved;
};
#pragma pack(pop)

// Physics/Vacuum: dense arrays (one float per tick)
// NumEntries floats, covering CENTER±range ticks
```

---

## 6. Debug Draw Visualization (Alternative to Niagara)

When Niagara configuration is complex or Remote Control can't set up User Parameters, use Debug Draw for immediate visualization:

```cpp
// MwtHeatmapRenderer.h
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MwtHeatmapRenderer.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class MYPROJECT_API UMwtHeatmapRenderer : public UActorComponent {
  GENERATED_BODY()

public:
  UMwtHeatmapRenderer();
  virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                              FActorComponentTickFunction* ThisTickFunction) override;

  // Scale settings
  UPROPERTY(EditAnywhere, Category = "Scale")
  float TickWorldScale = 5.0f;  // World units per price tick
  
  UPROPERTY(EditAnywhere, Category = "Scale")
  float TimeWorldScale = 20.0f; // World units per second
  
  UPROPERTY(EditAnywhere, Category = "Scale")
  int32 VisibleTickRange = 200; // ±100 ticks from center

  // Layer visibility
  UPROPERTY(EditAnywhere, Category = "Layers")
  bool bShowWall = true;
  
  UPROPERTY(EditAnywhere, Category = "Layers")
  bool bShowPhysics = true;

  // Data update methods
  void UpdateWallColumn(const TArray<float>& AskData, const TArray<float>& BidData);
  void UpdatePhysicsColumn(const TArray<float>& SignedData);
  void AdvanceTime();

private:
  static constexpr int32 LAYER_HEIGHT = 801;  // ±400 ticks
  static constexpr int32 CENTER_IDX = 400;
  
  TArray<FColor> WallData;
  TArray<FColor> PhysicsData;
  int32 HeadColumn = 0;
  int32 HistorySeconds = 120;
  
  int32 GetTextureIndex(int32 Row, int32 Col) const {
    return Row * HistorySeconds + Col;
  }
};
```

```cpp
// MwtHeatmapRenderer.cpp
#include "MwtHeatmapRenderer.h"
#include "DrawDebugHelpers.h"

void UMwtHeatmapRenderer::TickComponent(float DeltaTime, ELevelTick TickType,
                                         FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  
  if (!GetWorld() || !GetOwner()) return;
  
  FVector Origin = GetOwner()->GetActorLocation();
  
  // Tile sizes
  float TileWidth = TimeWorldScale;
  float TileHeight = TickWorldScale;
  
  int32 NumVisibleColumns = FMath::Min(HistorySeconds, 90);
  int32 HalfRange = VisibleTickRange / 2;
  
  for (int32 Col = 0; Col < NumVisibleColumns; Col++) {
    int32 DataCol = (HeadColumn - Col + HistorySeconds) % HistorySeconds;
    float XPos = -Col * TimeWorldScale;
    
    int32 StartRow = CENTER_IDX - HalfRange;
    int32 EndRow = CENTER_IDX + HalfRange;
    
    // Draw Wall layer
    if (bShowWall) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        FColor C = WallData[Idx];
        
        if (C.A > 5) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileWidth/2, 0, ZPos);
          FVector Extent(TileWidth/2, 3, TileHeight/2);
          
          FColor DrawColor;
          if (C.B > C.R) {
            // Ask (cyan/blue)
            DrawColor = FColor(0, C.B, C.B, 255);
          } else {
            // Bid (red)
            DrawColor = FColor(C.R, 0, 0, 255);
          }
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }
    
    // Draw Physics layer (behind)
    if (bShowPhysics) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        FColor C = PhysicsData[Idx];
        
        if (C.A > 5) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileWidth/2, -10, ZPos);
          FVector Extent(TileWidth/2, 2, TileHeight/2);
          
          FColor DrawColor;
          if (C.G > C.R) {
            DrawColor = FColor(0, C.G, 0, 255);  // Green = positive
          } else {
            DrawColor = FColor(C.R, 0, 0, 255);  // Red = negative
          }
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }
  }
  
  // Spot line at Z=0
  FVector LineStart = Origin + FVector(0, 15, 0);
  FVector LineEnd = Origin + FVector(-NumVisibleColumns * TimeWorldScale, 15, 0);
  DrawDebugLine(GetWorld(), LineStart, LineEnd, FColor::Cyan, false, -1.0f, 0, 5.0f);
}
```

**Advantages of Debug Draw**:
- No material/shader setup required
- Visible immediately in Play mode
- No Niagara configuration needed
- Easy to debug and iterate

**Disadvantages**:
- Not as performant as GPU-based rendering
- Limited visual effects
- Only visible in Play mode (not Editor viewport)

---

## 7. Config File Patterns

### DefaultEngine.ini - Set Default Map

```ini
[/Script/EngineSettings.GameMapsSettings]
EditorStartupMap=/Game/Maps/MWT_Main
GameDefaultMap=/Game/Maps/MWT_Main
```

### Cleanup Unwanted Maps via Filesystem

```bash
# Close UE Editor first!
cd "/path/to/UnrealProject/Content"
rm -f NewMap.umap NewMap1.umap Untitled.umap
rm -f NewMap_BuiltData.uasset  # Remove built data too
```

---

## 8. Data Flow Architecture

```
┌─────────────────┐
│  Backend API    │  WebSocket (Arrow IPC)
│  (Python/Fast)  │─────────────────────────┐
└─────────────────┘                         │
                                            ▼
                              ┌─────────────────────────┐
                              │     Python Bridge       │
                              │  - Decode Arrow IPC     │
                              │  - Transform data       │
                              │  - Pack UDP packets     │
                              └───────────┬─────────────┘
                                          │ UDP Port 7777
                                          ▼
                              ┌─────────────────────────┐
                              │   UE5 C++ Receiver      │
                              │  - Parse UDP packets    │
                              │  - Update arrays        │
                              │  - Apply dissipation    │
                              └───────────┬─────────────┘
                                          │
                                          ▼
                              ┌─────────────────────────┐
                              │   Heatmap Renderer      │
                              │  - Ring buffer storage  │
                              │  - Debug Draw / Niagara │
                              │  - Layer compositing    │
                              └─────────────────────────┘
```

### Source Data Schema (frontend_data.json)

The backend streams via WebSocket with Arrow IPC encoding:

```json
{
  "protocol": {
    "transport": "WebSocket",
    "endpoint": "ws://localhost:8000/v1/hud/stream",
    "format": "Arrow IPC",
    "cadence": "1 second windows"
  },
  "streams": {
    "snap": {
      "fields": ["window_end_ts_ns", "mid_price", "spot_ref_price_int", "book_valid"],
      "purpose": "Spot price anchor (cyan line)"
    },
    "wall": {
      "fields": ["rel_ticks", "side", "depth_qty_rest"],
      "purpose": "Liquidity heatmap. Blue=asks (above spot), Red=bids (below spot)"
    },
    "vacuum": {
      "fields": ["rel_ticks", "vacuum_score"],
      "purpose": "Black overlay showing liquidity erosion (0-1 score)"
    },
    "physics": {
      "fields": ["rel_ticks", "physics_score", "physics_score_signed"],
      "purpose": "Directional ease. Green=up ease (+), Red=down ease (-)"
    },
    "gex": {
      "fields": ["rel_ticks", "gex_abs", "gex", "gex_imbalance_ratio"],
      "purpose": "Gamma exposure. Green=calls, Red=puts"
    }
  },
  "constants": {
    "TICK_SIZE": 0.25,
    "TICK_INT": 250000000,
    "HUD_STREAM_MAX_TICKS": 400
  }
}
```

### Key Field Meanings
- `rel_ticks`: Ticks relative to spot. +N = above spot (asks), -N = below spot (bids)
- `side`: "A" (ask/blue) or "B" (bid/red)
- `depth_qty_rest`: Resting order depth (>500ms old). Frontend uses `log1p(depth)` for intensity
- `vacuum_score`: 0-1 normalized. Higher = more vacuum (liquidity eroding)
- `physics_score_signed`: +ve for upward ease, -ve for downward ease
- `gex_imbalance_ratio`: -1 to +1. Positive=call-heavy (green), Negative=put-heavy (red)

---

## 9. Tick-Native Coordinate System

The visualization uses a **tick-native** coordinate system where the spot price is always at center (Z=0):

```
Price Space (ticks)     World Space (UE units)
   +400 ticks      →    Z = +2000 (with scale 5.0)
   +100 ticks      →    Z = +500
   Spot (0)        →    Z = 0
   -100 ticks      →    Z = -500
   -400 ticks      →    Z = -2000

Time                    X Position
   Now (Head)      →    X = 0
   1 sec ago       →    X = -20 (with scale 20.0)
   60 sec ago      →    X = -1200
```

### Dissipation Model

```cpp
// Physics and Vacuum decay with τ = 5 seconds
float DecayFactor = FMath::Exp(-DeltaTime / 5.0f);

for (int32 i = 0; i < PhysicsData.Num(); i++) {
  FColor& C = PhysicsData[i];
  C.R = (uint8)(C.R * DecayFactor);
  C.G = (uint8)(C.G * DecayFactor);
  C.A = (uint8)(C.A * DecayFactor);
}

// Wall data has τ = 0 (instant clear on each update)
```

---

## 10. Remote Control Limitations

### What Remote Control CAN Do
- Get/set UPROPERTY values (with proper specifiers)
- Call UFUNCTION(BlueprintCallable) methods
- Search assets in Content Browser
- Load/save levels
- Spawn actors from Blueprint classes
- Query actor lists

### What Remote Control CANNOT Do
- Create new assets (Blueprints, Materials, Niagara Systems)
- Configure Niagara User Parameters
- Modify Blueprint graphs
- Create component hierarchies at runtime
- Access editor-only functionality in Play mode

### Workarounds
1. **Pre-create assets** in Editor, then manipulate via Remote Control
2. **Use C++ components** instead of Blueprint components for full control
3. **Filesystem operations** for asset deletion/copying (with Editor closed)
4. **DefaultEngine.ini** for configuration that persists

---

## 11. Blueprint Automation Workflow

When C++ class hierarchy changes, Blueprints must be recreated:

```python
# 1. Delete old Blueprint via filesystem (Editor closed)
import os
bp_path = "/path/to/UnrealProject/Content/Blueprints/BP_MyActor.uasset"
if os.path.exists(bp_path):
    os.remove(bp_path)

# 2. Open Editor - it will recompile C++ on startup

# 3. Via Remote Control, spawn the C++ class directly (if no BP needed)
client.call_function(
    "/Script/UnrealEd.Default__EditorLevelLibrary",
    "SpawnActorFromClass",
    {
        "ActorClass": {"ObjectPath": "/Script/MyProject.MwtUdpReceiver"},
        "Location": {"X": 0, "Y": 0, "Z": 0}
    }
)

# 4. Save level
client.call_function(
    "/Script/UnrealEd.Default__EditorLevelLibrary",
    "SaveCurrentLevel",
    {}
)
```

---

## 12. Debugging Checklist

### Data Not Appearing
1. Check backend is running: `curl http://localhost:8000/health`
2. Check bridge is running: look for "Connected to WebSocket" log
3. Check UDP receiver: UE log should show "Listening on port 7777"
4. Check packet reception: enable `bLogPackets = true`

### Visualization Wrong Scale/Position
1. Verify `TickWorldScale` and `TimeWorldScale` values
2. Check camera position relative to Origin
3. Enable grid lines to see coordinate system
4. Log spot reference tick value

### Crashes on Play
1. Check Output Log for specific error
2. Common: `FObjectFinder` outside constructor
3. Common: Null pointer on component access
4. Force rebuild: delete `Binaries/` and `Intermediate/`

### Remote Control Not Responding
1. Verify plugin enabled: Edit > Plugins > Remote Control API
2. Check port: default is 30010
3. Test: `curl http://localhost:30010/remote/info`
4. Firewall may block non-localhost connections

---

## 13. File Structure

```
UnrealProject/
├── Config/
│   └── DefaultEngine.ini        # Map settings
├── Content/
│   ├── Maps/
│   │   └── MWT_Main.umap       # Main level
│   └── Blueprints/
│       └── BP_MwtReceiver.uasset
├── Source/
│   └── MyProject/
│       ├── MyProject.Build.cs   # Module dependencies
│       ├── MwtUdpReceiver.h
│       ├── MwtUdpReceiver.cpp
│       ├── MwtHeatmapRenderer.h
│       └── MwtHeatmapRenderer.cpp
├── Binaries/                    # DELETE for clean rebuild
├── Intermediate/                # DELETE for clean rebuild
└── DerivedDataCache/            # DELETE for clean rebuild
```

---

## Summary

This skill enables Claude to:
1. **Connect** to UE5 via Remote Control API (Python client)
2. **Develop** C++ actors and components with proper UPROPERTY/UFUNCTION exposure
3. **Stream data** via UDP with efficient binary protocols
4. **Visualize** data using Debug Draw or Niagara
5. **Debug** common issues with systematic approaches
6. **Manage** assets and builds via filesystem when needed

Key principles:
- Always verify which code version is actually compiled
- Use C++ for full control, Blueprints for configurability
- Debug Draw is faster to iterate than Niagara
- Clean rebuilds solve many mysterious issues
- Remote Control is powerful but has clear limitations
