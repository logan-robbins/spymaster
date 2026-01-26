#include "MwtUdpReceiver.h"
#include "MwtHeatmapRenderer.h"
#include "Common/UdpSocketBuilder.h"
#include "SocketSubsystem.h"

AMwtUdpReceiver::AMwtUdpReceiver() {
  PrimaryActorTick.bCanEverTick = true;

  // Create heatmap renderer component
  HeatmapRenderer = CreateDefaultSubobject<UMwtHeatmapRenderer>(TEXT("HeatmapRenderer"));
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
                     .WithReceiveBufferSize(2 * 1024 * 1024)
                     .Build();

  if (ListenSocket) {
    UDPReceiver = new FUdpSocketReceiver(ListenSocket, FTimespan::FromMilliseconds(1),
                                         TEXT("MwtUdpReceiverThread"));
    UDPReceiver->OnDataReceived().BindUObject(this, &AMwtUdpReceiver::OnDataReceived);
    UDPReceiver->Start();
    UE_LOG(LogTemp, Log, TEXT("MWT UDP Receiver: Listening on port %d"), Port);
  } else {
    UE_LOG(LogTemp, Error, TEXT("MWT UDP Receiver: Failed to create socket on port %d"), Port);
  }

  LastWindowAdvanceTime = GetWorld()->GetTimeSeconds();
}

void AMwtUdpReceiver::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (UDPReceiver) {
    UDPReceiver->Stop();
    delete UDPReceiver;
    UDPReceiver = nullptr;
  }

  if (ListenSocket) {
    ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ListenSocket);
    ListenSocket = nullptr;
  }

  Super::EndPlay(EndPlayReason);
}

void AMwtUdpReceiver::InitArrays() {
  WallAsk.Init(0.0f, LAYER_HEIGHT);
  WallBid.Init(0.0f, LAYER_HEIGHT);
  Vacuum.Init(0.0f, LAYER_HEIGHT);
  PhysicsSigned.Init(0.0f, LAYER_HEIGHT);
  GexAbs.Init(0.0f, LAYER_HEIGHT);
  GexImbalance.Init(0.0f, LAYER_HEIGHT);
}

void AMwtUdpReceiver::ResetState() {
  FScopeLock Lock(&DataMutex);
  
  for (int32 i = 0; i < LAYER_HEIGHT; i++) {
    WallAsk[i] = 0.0f;
    WallBid[i] = 0.0f;
    Vacuum[i] = 0.0f;
    PhysicsSigned[i] = 0.0f;
    GexAbs[i] = 0.0f;
    GexImbalance[i] = 0.0f;
  }

  CurrentWindowTs = 0;
  SpotRefPriceInt = 0;
  SpotRefTick = 0.0f;
  MidPrice = 0.0;
  bBookValid = false;
  PacketsReceived = 0;
  bDataDirty = true;

  UE_LOG(LogTemp, Log, TEXT("MWT: State reset"));
}

void AMwtUdpReceiver::OnDataReceived(const FArrayReaderPtr& ArrayReaderPtr,
                                      const FIPv4Endpoint& Endpoint) {
  TArray<uint8> Data;
  Data.Append(ArrayReaderPtr->GetData(), ArrayReaderPtr->Num());
  ProcessPacket(Data);
}

void AMwtUdpReceiver::ProcessPacket(const TArray<uint8>& Data) {
  if (Data.Num() < sizeof(FMwtPacketHeader)) {
    return;
  }

  const FMwtPacketHeader* Header = reinterpret_cast<const FMwtPacketHeader*>(Data.GetData());

  // Validate magic
  if (FMemory::Memcmp(Header->Magic, "MWT1", 4) != 0) {
    if (bLogPackets) {
      UE_LOG(LogTemp, Warning, TEXT("MWT: Invalid magic in packet"));
    }
    return;
  }

  // Validate version
  if (Header->Version != 1) {
    if (bLogPackets) {
      UE_LOG(LogTemp, Warning, TEXT("MWT: Unknown version %d"), Header->Version);
    }
    return;
  }

  FScopeLock Lock(&DataMutex);

  PacketsReceived++;
  LastSurfaceId = Header->SurfaceId;

  // Update spot reference
  if (Header->SpotRefPriceInt > 0) {
    SpotRefPriceInt = Header->SpotRefPriceInt;
    SpotRefTick = static_cast<float>(SpotRefPriceInt / TICK_INT);
  }

  // Check for new window (triggers decay and ring buffer advance)
  if (Header->WindowEndTsNs > CurrentWindowTs) {
    CurrentWindowTs = Header->WindowEndTsNs;

    // Wall has τ=0, clear immediately
    ClearWallArrays();

    // Apply exponential decay to Vacuum and Physics (τ=5s)
    // Since windows are ~1 second apart: decay = exp(-1/5) ≈ 0.8187
    ApplyDecay();

    // Advance heatmap renderer time column
    if (HeatmapRenderer) {
      HeatmapRenderer->AdvanceTime();
      HeatmapRenderer->UpdateSpotRef(SpotRefTick);
    }

    if (bLogSurfaceUpdates) {
      UE_LOG(LogTemp, Log, TEXT("MWT: New window ts=%lld, spot=%.2f ticks"),
             CurrentWindowTs, SpotRefTick);
    }
  }

  // Process payload based on surface type
  const uint8* PayloadStart = Data.GetData() + sizeof(FMwtPacketHeader);
  int32 PayloadSize = Data.Num() - sizeof(FMwtPacketHeader);

  switch (Header->SurfaceId) {
    case 1: // SNAP
      ProcessSnapPayload(PayloadStart, PayloadSize, Header->Count);
      break;
    case 2: // WALL
      ProcessWallPayload(PayloadStart, PayloadSize, Header->Count);
      break;
    case 3: // VACUUM
      ProcessVacuumPayload(PayloadStart, PayloadSize, Header->Count);
      break;
    case 4: // PHYSICS
      ProcessPhysicsPayload(PayloadStart, PayloadSize, Header->Count);
      break;
    case 5: // GEX
      ProcessGexPayload(PayloadStart, PayloadSize, Header->Count);
      break;
    default:
      if (bLogPackets) {
        UE_LOG(LogTemp, Warning, TEXT("MWT: Unknown surface ID %d"), Header->SurfaceId);
      }
      break;
  }

  bDataDirty = true;
}

void AMwtUdpReceiver::ProcessSnapPayload(const uint8* Data, int32 Size, uint32 Count) {
  if (Size < sizeof(FMwtSnapEntry) || Count < 1) return;

  const FMwtSnapEntry* Entry = reinterpret_cast<const FMwtSnapEntry*>(Data);
  MidPrice = Entry->MidPrice;
  bBookValid = Entry->bBookValid;

  if (bLogSurfaceUpdates) {
    UE_LOG(LogTemp, Verbose, TEXT("MWT SNAP: mid=%.4f, valid=%d"), MidPrice, bBookValid);
  }
}

void AMwtUdpReceiver::ProcessWallPayload(const uint8* Data, int32 Size, uint32 Count) {
  int32 EntrySize = sizeof(FMwtWallEntry);
  int32 MaxEntries = Size / EntrySize;
  uint32 NumEntries = FMath::Min(Count, (uint32)MaxEntries);

  for (uint32 i = 0; i < NumEntries; i++) {
    const FMwtWallEntry* Entry = reinterpret_cast<const FMwtWallEntry*>(Data + i * EntrySize);
    
    int32 Idx = CENTER_IDX + Entry->RelTicks;
    if (Idx >= 0 && Idx < LAYER_HEIGHT) {
      if (Entry->Side == 1) { // Ask
        WallAsk[Idx] = Entry->WallIntensity;
      } else { // Bid
        WallBid[Idx] = Entry->WallIntensity;
      }
    }
  }

  if (bLogSurfaceUpdates) {
    UE_LOG(LogTemp, Verbose, TEXT("MWT WALL: %d entries"), NumEntries);
  }
}

void AMwtUdpReceiver::ProcessVacuumPayload(const uint8* Data, int32 Size, uint32 Count) {
  int32 EntrySize = sizeof(FMwtVacuumEntry);
  int32 MaxEntries = Size / EntrySize;
  uint32 NumEntries = FMath::Min(Count, (uint32)MaxEntries);

  for (uint32 i = 0; i < NumEntries; i++) {
    const FMwtVacuumEntry* Entry = reinterpret_cast<const FMwtVacuumEntry*>(Data + i * EntrySize);
    
    int32 Idx = CENTER_IDX + Entry->RelTicks;
    if (Idx >= 0 && Idx < LAYER_HEIGHT) {
      Vacuum[Idx] = Entry->VacuumScore;
    }
  }

  if (bLogSurfaceUpdates) {
    UE_LOG(LogTemp, Verbose, TEXT("MWT VACUUM: %d entries"), NumEntries);
  }
}

void AMwtUdpReceiver::ProcessPhysicsPayload(const uint8* Data, int32 Size, uint32 Count) {
  int32 EntrySize = sizeof(FMwtPhysicsEntry);
  int32 MaxEntries = Size / EntrySize;
  uint32 NumEntries = FMath::Min(Count, (uint32)MaxEntries);

  for (uint32 i = 0; i < NumEntries; i++) {
    const FMwtPhysicsEntry* Entry = reinterpret_cast<const FMwtPhysicsEntry*>(Data + i * EntrySize);
    
    int32 Idx = CENTER_IDX + Entry->RelTicks;
    if (Idx >= 0 && Idx < LAYER_HEIGHT) {
      PhysicsSigned[Idx] = Entry->PhysicsScoreSigned;
    }
  }

  if (bLogSurfaceUpdates) {
    UE_LOG(LogTemp, Verbose, TEXT("MWT PHYSICS: %d entries"), NumEntries);
  }
}

void AMwtUdpReceiver::ProcessGexPayload(const uint8* Data, int32 Size, uint32 Count) {
  int32 EntrySize = sizeof(FMwtGexEntry);
  int32 MaxEntries = Size / EntrySize;
  uint32 NumEntries = FMath::Min(Count, (uint32)MaxEntries);

  for (uint32 i = 0; i < NumEntries; i++) {
    const FMwtGexEntry* Entry = reinterpret_cast<const FMwtGexEntry*>(Data + i * EntrySize);
    
    int32 Idx = CENTER_IDX + Entry->RelTicks;
    if (Idx >= 0 && Idx < LAYER_HEIGHT) {
      GexAbs[Idx] = Entry->GexAbs;
      GexImbalance[Idx] = Entry->ImbalanceRatio;
    }
  }

  if (bLogSurfaceUpdates) {
    UE_LOG(LogTemp, Verbose, TEXT("MWT GEX: %d entries"), NumEntries);
  }
}

void AMwtUdpReceiver::ClearWallArrays() {
  for (int32 i = 0; i < LAYER_HEIGHT; i++) {
    WallAsk[i] = 0.0f;
    WallBid[i] = 0.0f;
  }
}

void AMwtUdpReceiver::ApplyDecay() {
  // Decay factor for τ=5s at 1Hz update rate
  // exp(-1/5) = 0.8187
  const float Decay = 0.8187f;

  for (int32 i = 0; i < LAYER_HEIGHT; i++) {
    Vacuum[i] *= Decay;
    PhysicsSigned[i] *= Decay;
  }
}

void AMwtUdpReceiver::Tick(float DeltaTime) {
  Super::Tick(DeltaTime);

  // Push data to renderer when dirty
  if (bDataDirty) {
    PushToRenderer();
    bDataDirty = false;
  }
}

void AMwtUdpReceiver::PushToRenderer() {
  if (!HeatmapRenderer) return;

  FScopeLock Lock(&DataMutex);

  // Push all surface data to heatmap renderer
  HeatmapRenderer->UpdateWallColumn(WallAsk, WallBid);
  HeatmapRenderer->UpdateVacuumColumn(Vacuum);
  HeatmapRenderer->UpdatePhysicsColumn(PhysicsSigned);
  HeatmapRenderer->UpdateGexColumn(GexAbs, GexImbalance);
}
