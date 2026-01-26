#pragma once

// clang-format off
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Networking.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"
#include "NiagaraSystem.h"
#include "NiagaraComponent.h"
#include "MwtUdpReceiver.generated.h"
// clang-format on

// Packet Structs (Must match Python pack format)
#pragma pack(push, 1)

struct FMwtPacketHeader {
  char Magic[4]; // "MWT1"
  uint16 Version;
  uint16 SurfaceId; // 1=SNAP, 2=WALL, 3=VACUUM, 4=PHYSICS, 5=GEX
  int64 WindowEndTsNs;
  int64 SpotRefPriceInt;
  uint32 Count;
  uint32 Flags;
  int64 PredictionHorizonNs;
};

struct FMwtSnapEntry {
  double MidPrice;
  bool bBookValid;
  uint8 Padding[7];
};

struct FMwtWallEntry {
  int16 RelTicks;      // [-400, +400]
  uint8 Side;          // 0=Bid, 1=Ask
  float WallIntensity; // 0..1
  float WallErosion;   // 0..1
};

struct FMwtVacuumEntry {
  int16 RelTicks;
  float VacuumScore;
  float Turbulence;
};

struct FMwtPhysicsEntry {
  int16 RelTicks;
  float PhysicsScoreSigned;
};

struct FMwtGexEntry {
  int16 RelTicks;
  float GexAbs;
  float ImbalanceRatio;
};

#pragma pack(pop)

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

  // UDP
  FSocket *ListenSocket;
  FUdpSocketReceiver *UDPReceiver;
  void OnDataReceived(const FArrayReaderPtr &ArrayReaderPtr,
                      const FIPv4Endpoint &Endpoint);

  // Niagara
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  UNiagaraComponent *NiagaraComp;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  int32 Port = 7777;

  // Data State
  FCriticalSection DataMutex;

  // Arrays for Niagara (Size 801)
  TArray<float> WallAsk;
  TArray<float> WallBid;
  TArray<float> WallErosion;
  TArray<float> Vacuum;
  TArray<float> PhysicsSigned;
  TArray<float> GexAbs;
  TArray<float> GexImbalance;

  // Current State
  int64 CurrentWindowTs;

private:
  void ProcessPacket(const TArray<uint8> &Data);
  void InitArrays();
  void UpdateNiagara();
};
