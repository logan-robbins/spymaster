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
  WallAsk.Init(0.0f, Size);
  WallBid.Init(0.0f, Size);
  WallErosion.Init(0.0f, Size);
  Vacuum.Init(0.0f, Size);
  PhysicsSigned.Init(0.0f, Size);
  GexAbs.Init(0.0f, Size);
  GexImbalance.Init(0.0f, Size);
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
  if (Header->Version != 1)
    return;

  FScopeLock Lock(&DataMutex);

  // Check for new window -> Clear / Decay
  if (Header->WindowEndTsNs > CurrentWindowTs) {
    CurrentWindowTs = Header->WindowEndTsNs;

    // Clear Wall (Immediate)
    float ClearVal = 0.0f;
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

    if (Header->SurfaceId == 2) // WALL
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
      const FMwtVacuumEntry *Entry =
          reinterpret_cast<const FMwtVacuumEntry *>(Data.GetData() + Offset);
      Offset += sizeof(FMwtVacuumEntry);

      int32 Idx = CenterIdx + Entry->RelTicks;
      if (Idx >= 0 && Idx < 801) {
        Vacuum[Idx] = Entry->VacuumScore;
      }
    } else if (Header->SurfaceId == 4) // PHYSICS
    {
      const FMwtPhysicsEntry *Entry =
          reinterpret_cast<const FMwtPhysicsEntry *>(Data.GetData() + Offset);
      Offset += sizeof(FMwtPhysicsEntry);

      int32 Idx = CenterIdx + Entry->RelTicks;
      if (Idx >= 0 && Idx < 801) {
        PhysicsSigned[Idx] = Entry->PhysicsScoreSigned;
      }
    } else if (Header->SurfaceId == 5) // GEX
    {
      const FMwtGexEntry *Entry =
          reinterpret_cast<const FMwtGexEntry *>(Data.GetData() + Offset);
      Offset += sizeof(FMwtGexEntry);

      int32 Idx = CenterIdx + Entry->RelTicks;
      if (Idx >= 0 && Idx < 801) {
        GexAbs[Idx] = Entry->GexAbs;
        GexImbalance[Idx] = Entry->ImbalanceRatio;
      }
    } else {
      // Skip unknown payload? We can't know size.
      // But we know SNAP is size 16.
      if (Header->SurfaceId == 1)
        Offset += sizeof(FMwtSnapEntry);
      else
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

  // Push Arrays to User Parameters
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.WallAsk"), WallAsk);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.WallBid"), WallBid);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.WallErosion"),
                                            WallErosion);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.Vacuum"), Vacuum);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.PhysicsSigned"),
                                            PhysicsSigned);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.GexAbs"), GexAbs);
  NiagaraComp->SetNiagaraVariableFloatArray(TEXT("User.GexImbalance"),
                                            GexImbalance);
}
