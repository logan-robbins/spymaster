#include "MwtUdpReceiver.h"
#include "Common/UdpSocketBuilder.h"
#include "SocketSubsystem.h"

AMwtUdpReceiver::AMwtUdpReceiver() {
  PrimaryActorTick.bCanEverTick = true;

  // Default Niagara component (user should attach in BP)
  NiagaraComp =
      CreateDefaultSubobject<UNiagaraComponent>(TEXT("NiagaraSystem"));
  RootComponent = NiagaraComp;

  CurrentWindowTs = 0;
}

void AMwtUdpReceiver::BeginPlay() {
  Super::BeginPlay();

  InitArrays();
  ResetSurfaces();

  // Start UDP Receiver
  FIPv4Endpoint Endpoint(FIPv4Address::Any, Port);
  ListenSocket = FUdpSocketBuilder(TEXT("MwtUdpSocket"))
                     .AsNonBlocking()
                     .AsReusable()
                     .BoundToEndpoint(Endpoint)
                     .ReceiveBufferSize(2 * 1024 * 1024)
                     .Build();

  if (ListenSocket) {
    UDPReceiver =
        new FUdpSocketReceiver(ListenSocket, FTimespan::FromMilliseconds(1),
                               TEXT("MwtUdpReceiverThread"));
    UDPReceiver->OnDataReceived().BindUObject(this,
                                              &AMwtUdpReceiver::OnDataReceived);
    UDPReceiver->Start();
    UE_LOG(LogTemp, Log, TEXT("MWT UDP Listening on Port %d"), Port);
  }
}

void AMwtUdpReceiver::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (UDPReceiver) {
    UDPReceiver->Stop();
    delete UDPReceiver;
    UDPReceiver = nullptr;
  }

  if (ListenSocket) {
    ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)
        ->DestroySocket(ListenSocket);
    ListenSocket = nullptr;
  }

  Super::EndPlay(EndPlayReason);
}

void AMwtUdpReceiver::InitArrays() {
  // 801 ticks (+/- 400 around spot)
  int32 Size = 801;
  int32 HistorySize = FMath::Max(HistorySeconds, 1);
  WallAsk.Init(0.0f, Size);
  WallBid.Init(0.0f, Size);
  WallErosion.Init(0.0f, Size);
  Vacuum.Init(0.0f, Size);
  PhysicsSigned.Init(0.0f, Size);
  GexAbs.Init(0.0f, Size);
  GexImbalance.Init(0.0f, Size);
  SpotHistory.Init(0.0f, HistorySize);
  SpotHead = 0;

  WallAskScaled.Init(0.0f, Size);
  WallBidScaled.Init(0.0f, Size);
  WallErosionScaled.Init(0.0f, Size);
  VacuumScaled.Init(0.0f, Size);
  PhysicsScaled.Init(0.0f, Size);
  GexAbsScaled.Init(0.0f, Size);
  GexImbalanceScaled.Init(0.0f, Size);
}

void AMwtUdpReceiver::ResetSurfaces() {
  FScopeLock Lock(&DataMutex);
  FMemory::Memset(WallAsk.GetData(), 0, WallAsk.Num() * sizeof(float));
  FMemory::Memset(WallBid.GetData(), 0, WallBid.Num() * sizeof(float));
  FMemory::Memset(WallErosion.GetData(), 0, WallErosion.Num() * sizeof(float));
  FMemory::Memset(Vacuum.GetData(), 0, Vacuum.Num() * sizeof(float));
  FMemory::Memset(PhysicsSigned.GetData(), 0,
                  PhysicsSigned.Num() * sizeof(float));
  FMemory::Memset(GexAbs.GetData(), 0, GexAbs.Num() * sizeof(float));
  FMemory::Memset(GexImbalance.GetData(), 0,
                  GexImbalance.Num() * sizeof(float));
  FMemory::Memset(SpotHistory.GetData(), 0,
                  SpotHistory.Num() * sizeof(float));
  SpotHead = 0;
  CurrentWindowTs = 0;
  LastSpotTs = 0;
  SpotRefPriceInt = 0;
  SpotRefTick = 0.0f;
  MidPrice = 0.0;
  bBookValid = false;
}

void AMwtUdpReceiver::SetLayerGains(float InWallGain, float InVacuumGain,
                                    float InPhysicsGain, float InGexGain) {
  FScopeLock Lock(&DataMutex);
  WallGain = InWallGain;
  VacuumGain = InVacuumGain;
  PhysicsGain = InPhysicsGain;
  GexGain = InGexGain;
}

void AMwtUdpReceiver::SetLayerEnabled(bool bWall, bool bVacuum, bool bPhysics,
                                      bool bGex, bool bSpotLine) {
  FScopeLock Lock(&DataMutex);
  bEnableWall = bWall;
  bEnableVacuum = bVacuum;
  bEnablePhysics = bPhysics;
  bEnableGex = bGex;
  bEnableSpotLine = bSpotLine;
}

void AMwtUdpReceiver::OnDataReceived(const FArrayReaderPtr &ArrayReaderPtr,
                                     const FIPv4Endpoint &Endpoint) {
  TArray<uint8> Data = *ArrayReaderPtr;
  ProcessPacket(Data);
}

void AMwtUdpReceiver::ProcessPacket(const TArray<uint8> &Data) {
  if (Data.Num() < sizeof(FMwtPacketHeader))
    return;

  const FMwtPacketHeader *Header =
      reinterpret_cast<const FMwtPacketHeader *>(Data.GetData());

  // Basic validation
  if (FMemory::Memcmp(Header->Magic, "MWT1", 4) != 0)
    return;
  if (Header->Version != 1)
    return;

  FScopeLock Lock(&DataMutex);

  SpotRefPriceInt = Header->SpotRefPriceInt;
  if (SpotRefPriceInt > 0) {
    const double TickInt = 250000000.0;
    SpotRefTick = static_cast<float>(SpotRefPriceInt / TickInt);
  } else {
    SpotRefTick = 0.0f;
  }

  // Check for new window -> Clear / Decay
  if (Header->WindowEndTsNs > CurrentWindowTs) {
    CurrentWindowTs = Header->WindowEndTsNs;

    // Clear Wall (Immediate)
    FMemory::Memset(WallAsk.GetData(), 0, WallAsk.Num() * sizeof(float));
    FMemory::Memset(WallBid.GetData(), 0, WallBid.Num() * sizeof(float));
    FMemory::Memset(WallErosion.GetData(), 0,
                    WallErosion.Num() * sizeof(float));

    // Decay Vacuum / Physics (Simple exponential decay approx 0.82 for 1s if
    // tau=5s? Frontend says: new_cell = old_cell * exp(-dt/tau). Since we
    // receive sparse updates, we should decay everything ONCE per window tick.
    // Assuming 1Hz packets: exp(-1/5) = 0.8187.
    float Decay = 0.8187f;

    for (float &Val : Vacuum)
      Val *= Decay;
    for (float &Val : PhysicsSigned)
      Val *= Decay;
    // GEX: preserve or decay? Spec says "no decay or slow". Let's preserve.
  }

  // Process payload
  int32 Offset = sizeof(FMwtPacketHeader);
  int32 CenterIdx = 400; // rel_ticks=0 is index 400

  for (uint32 i = 0; i < Header->Count; i++) {
    if (Offset >= Data.Num())
      break;

    if (Header->SurfaceId == 1) // SNAP
    {
      if (Offset + sizeof(FMwtSnapEntry) > Data.Num())
        break;
      const FMwtSnapEntry *Entry =
          reinterpret_cast<const FMwtSnapEntry *>(Data.GetData() + Offset);
      Offset += sizeof(FMwtSnapEntry);

      MidPrice = Entry->MidPrice;
      bBookValid = Entry->bBookValid;

      if (Header->WindowEndTsNs > LastSpotTs && SpotHistory.Num() > 0) {
        SpotHistory[SpotHead] = SpotRefTick;
        SpotHead = (SpotHead + 1) % SpotHistory.Num();
        LastSpotTs = Header->WindowEndTsNs;
      }
    } else if (Header->SurfaceId == 2) // WALL
    {
      if (Offset + sizeof(FMwtWallEntry) > Data.Num())
        break;
      const FMwtWallEntry *Entry =
          reinterpret_cast<const FMwtWallEntry *>(Data.GetData() + Offset);
      Offset += sizeof(FMwtWallEntry);

      int32 Idx = CenterIdx + Entry->RelTicks;
      if (Idx >= 0 && Idx < 801) {
        if (Entry->Side == 1) // Ask
          WallAsk[Idx] = Entry->WallIntensity;
        else // Bid
          WallBid[Idx] = Entry->WallIntensity;

        WallErosion[Idx] = Entry->WallErosion;
      }
    } else if (Header->SurfaceId == 3) // VACUUM
    {
      if (Offset + sizeof(FMwtVacuumEntry) > Data.Num())
        break;
      const FMwtVacuumEntry *Entry =
          reinterpret_cast<const FMwtVacuumEntry *>(Data.GetData() + Offset);
      Offset += sizeof(FMwtVacuumEntry);

      int32 Idx = CenterIdx + Entry->RelTicks;
      if (Idx >= 0 && Idx < 801) {
        Vacuum[Idx] = Entry->VacuumScore;
      }
    } else if (Header->SurfaceId == 4) // PHYSICS
    {
      if (Offset + sizeof(FMwtPhysicsEntry) > Data.Num())
        break;
      const FMwtPhysicsEntry *Entry =
          reinterpret_cast<const FMwtPhysicsEntry *>(Data.GetData() + Offset);
      Offset += sizeof(FMwtPhysicsEntry);

      int32 Idx = CenterIdx + Entry->RelTicks;
      if (Idx >= 0 && Idx < 801) {
        PhysicsSigned[Idx] = Entry->PhysicsScoreSigned;
      }
    } else if (Header->SurfaceId == 5) // GEX
    {
      if (Offset + sizeof(FMwtGexEntry) > Data.Num())
        break;
      const FMwtGexEntry *Entry =
          reinterpret_cast<const FMwtGexEntry *>(Data.GetData() + Offset);
      Offset += sizeof(FMwtGexEntry);

      int32 Idx = CenterIdx + Entry->RelTicks;
      if (Idx >= 0 && Idx < 801) {
        GexAbs[Idx] = Entry->GexAbs;
        GexImbalance[Idx] = Entry->ImbalanceRatio;
      }
    } else {
      break;
    }
  }
}

void AMwtUdpReceiver::Tick(float DeltaTime) {
  Super::Tick(DeltaTime);

  UpdateNiagara();
}

void AMwtUdpReceiver::UpdateNiagara() {
  if (!NiagaraComp)
    return;

  FScopeLock Lock(&DataMutex);

  const float WallScale = bEnableWall ? WallGain : 0.0f;
  const float VacuumScale = bEnableVacuum ? VacuumGain : 0.0f;
  const float PhysicsScale = bEnablePhysics ? PhysicsGain : 0.0f;
  const float GexScale = bEnableGex ? GexGain : 0.0f;

  for (int32 i = 0; i < WallAsk.Num(); i++) {
    WallAskScaled[i] = WallAsk[i] * WallScale;
    WallBidScaled[i] = WallBid[i] * WallScale;
    WallErosionScaled[i] = WallErosion[i] * WallScale;
    VacuumScaled[i] = Vacuum[i] * VacuumScale;
    PhysicsScaled[i] = PhysicsSigned[i] * PhysicsScale;
    GexAbsScaled[i] = GexAbs[i] * GexScale;
    GexImbalanceScaled[i] = bEnableGex ? GexImbalance[i] : 0.0f;
  }

  // Push Arrays to User Parameters
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.WallAsk"),
                                            WallAskScaled);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.WallBid"),
                                            WallBidScaled);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.WallErosion"),
                                            WallErosionScaled);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.Vacuum"), VacuumScaled);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.PhysicsSigned"),
                                            PhysicsScaled);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.GexAbs"), GexAbsScaled);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.GexImbalance"),
                                            GexImbalanceScaled);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.SpotHistory"),
                                            SpotHistory);

  NiagaraComp->SetNiagaraVariableFloat(TEXT("User.MidPrice"),
                                       static_cast<float>(MidPrice));
  NiagaraComp->SetNiagaraVariableFloat(TEXT("User.SpotRefTick"), SpotRefTick);
  NiagaraComp->SetNiagaraVariableBool(TEXT("User.BookValid"), bBookValid);
  NiagaraComp->SetNiagaraVariableBool(TEXT("User.SpotLineEnabled"),
                                      bEnableSpotLine);
  NiagaraComp->SetNiagaraVariableInt(TEXT("User.SpotHead"), SpotHead);
  NiagaraComp->SetNiagaraVariableInt(TEXT("User.HistorySeconds"),
                                     SpotHistory.Num());
}
