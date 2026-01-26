#pragma once

// clang-format off
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Networking.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"
#include "MwtUdpReceiver.generated.h"
// clang-format on

// Forward declaration
class UMwtHeatmapRenderer;

// Packet Structs (Must match Python pack format - MWT-UDP v1 Protocol)
#pragma pack(push, 1)

struct FMwtPacketHeader {
  char Magic[4];              // "MWT1"
  uint16 Version;             // 1
  uint16 SurfaceId;           // 1=SNAP, 2=WALL, 3=VACUUM, 4=PHYSICS, 5=GEX
  int64 WindowEndTsNs;        // Canonical window timestamp
  int64 SpotRefPriceInt;      // Tick-aligned spot anchor
  uint32 Count;               // Number of entries
  uint32 Flags;               // Reserved (0 in v1)
  int64 PredictionHorizonNs;  // Reserved for future
};

struct FMwtSnapEntry {
  double MidPrice;
  bool bBookValid;
  uint8 Padding[7];
};

struct FMwtWallEntry {
  int16 RelTicks;       // [-400, +400]
  uint8 Side;           // 0=Bid, 1=Ask
  float WallIntensity;  // 0..1 (log1p normalized)
  float WallErosion;    // 0..1
};

struct FMwtVacuumEntry {
  int16 RelTicks;
  float VacuumScore;    // 0..1
  float Turbulence;
};

struct FMwtPhysicsEntry {
  int16 RelTicks;
  float PhysicsScoreSigned;  // +ve = up ease, -ve = down ease
};

struct FMwtGexEntry {
  int16 RelTicks;       // Must be multiple of 20 (ES $5 strikes)
  float GexAbs;         // Absolute gamma exposure
  float ImbalanceRatio; // [-1, +1] calls vs puts
};

#pragma pack(pop)

/**
 * AMwtUdpReceiver - Market Wind Tunnel Data Receiver
 * 
 * Receives UDP packets from the Python bridge and feeds them
 * to the heatmap visualization renderer.
 * 
 * Data Flow:
 *   Backend HUD Stream (WebSocket) 
 *   → Bridge (Python, Arrow IPC → UDP)
 *   → This Receiver (UDP → Arrays)
 *   → MwtHeatmapRenderer (Arrays → Visualization)
 */
UCLASS()
class MARKETWINDTUNNEL_API AMwtUdpReceiver : public AActor {
  GENERATED_BODY()

public:
  AMwtUdpReceiver();

protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
  virtual void Tick(float DeltaTime) override;

  // ============== Configuration ==============
  
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Network")
  int32 Port = 7777;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Debug")
  bool bLogPackets = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Debug")
  bool bLogSurfaceUpdates = true;

  // ============== Heatmap Renderer ==============
  
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT|Visualization")
  UMwtHeatmapRenderer* HeatmapRenderer;

  // ============== State (Read-Only) ==============
  
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT|State")
  double MidPrice = 0.0;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT|State")
  float SpotRefTick = 0.0f;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT|State")
  bool bBookValid = false;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT|State")
  int32 PacketsReceived = 0;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT|State")
  int32 LastSurfaceId = 0;

  // ============== Blueprint Functions ==============
  
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void ResetState();

private:
  // UDP Socket
  FSocket* ListenSocket = nullptr;
  FUdpSocketReceiver* UDPReceiver = nullptr;

  // Current window state
  int64 CurrentWindowTs = 0;
  int64 SpotRefPriceInt = 0;
  float LastWindowAdvanceTime = 0.0f;

  // Data arrays (801 elements, ±400 ticks from spot)
  static constexpr int32 LAYER_HEIGHT = 801;
  static constexpr int32 CENTER_IDX = 400;
  static constexpr double TICK_INT = 250000000.0;

  TArray<float> WallAsk;
  TArray<float> WallBid;
  TArray<float> Vacuum;
  TArray<float> PhysicsSigned;
  TArray<float> GexAbs;
  TArray<float> GexImbalance;

  FCriticalSection DataMutex;
  bool bDataDirty = false;

  // Processing
  void OnDataReceived(const FArrayReaderPtr& ArrayReaderPtr, const FIPv4Endpoint& Endpoint);
  void ProcessPacket(const TArray<uint8>& Data);
  void ProcessSnapPayload(const uint8* Data, int32 Size, uint32 Count);
  void ProcessWallPayload(const uint8* Data, int32 Size, uint32 Count);
  void ProcessVacuumPayload(const uint8* Data, int32 Size, uint32 Count);
  void ProcessPhysicsPayload(const uint8* Data, int32 Size, uint32 Count);
  void ProcessGexPayload(const uint8* Data, int32 Size, uint32 Count);

  void InitArrays();
  void ClearWallArrays();
  void ApplyDecay();
  void PushToRenderer();
};
