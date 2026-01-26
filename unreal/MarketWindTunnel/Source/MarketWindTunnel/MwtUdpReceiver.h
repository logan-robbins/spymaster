#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MwtUdpReceiver.generated.h"
#include "Networking.h"
#include "NiagaraComponent.h"
#include "NiagaraSystem.h"

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

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  int32 HistorySeconds = 1800;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  float WallGain = 1.0f;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  float VacuumGain = 1.0f;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  float PhysicsGain = 1.0f;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  float GexGain = 1.0f;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  bool bEnableWall = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  bool bEnableVacuum = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  bool bEnablePhysics = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  bool bEnableGex = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT")
  bool bEnableSpotLine = true;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT")
  double MidPrice = 0.0;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT")
  float SpotRefTick = 0.0f;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MWT")
  bool bBookValid = false;

  UFUNCTION(BlueprintCallable, Category = "MWT")
  void ResetSurfaces();

  UFUNCTION(BlueprintCallable, Category = "MWT")
  void SetLayerGains(float InWallGain, float InVacuumGain, float InPhysicsGain,
                     float InGexGain);

  UFUNCTION(BlueprintCallable, Category = "MWT")
  void SetLayerEnabled(bool bWall, bool bVacuum, bool bPhysics, bool bGex,
                       bool bSpotLine);

  // Data State (Thread Safe via Queue or Render Thread updates)
  // For V1, we'll use a thread-safe MPMC queue or simple locking.
  // Given the 1Hz update, locking is fine.

  FCriticalSection DataMutex;

  // Arrays for Niagara (Size 801)
  TArray<float> WallAsk;
  TArray<float> WallBid;
  TArray<float> WallErosion;
  TArray<float> Vacuum;
  TArray<float> PhysicsSigned;
  TArray<float> GexAbs;
  TArray<float> GexImbalance;
  TArray<float> SpotHistory;
  int32 SpotHead = 0;

  TArray<float> WallAskScaled;
  TArray<float> WallBidScaled;
  TArray<float> WallErosionScaled;
  TArray<float> VacuumScaled;
  TArray<float> PhysicsScaled;
  TArray<float> GexAbsScaled;
  TArray<float> GexImbalanceScaled;

  // Current State
  int64 CurrentWindowTs;
  int64 LastSpotTs = 0;
  int64 SpotRefPriceInt = 0;

private:
  void ProcessPacket(const TArray<uint8> &Data);
  void InitArrays();
  void UpdateNiagara();
};
