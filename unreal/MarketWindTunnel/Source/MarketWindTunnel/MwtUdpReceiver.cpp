#include "MwtUdpReceiver.h"
#include "Common/UdpSocketBuilder.h"
#include "NiagaraDataInterfaceArrayFunctionLibrary.h"
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
                     .WithReceiveBufferSize(2 * 1024 * 1024)
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

#include "DrawDebugHelpers.h"

void AMwtUdpReceiver::Tick(float DeltaTime) {
  Super::Tick(DeltaTime);

  UpdateNiagara();

  // auto-debug visualization (Bookmap Style Heatmap)
  if (Port == 7777 && GetWorld()) {
    FScopeLock Lock(&DataMutex);
    FVector ActorLoc = GetActorLocation();

    // Settings for Heatmap
    float TileHeight = 10.0f; // Height of one price tick
    float TileWidth = 500.0f; // Width of the chart (visual only)

    // Draw Ask Wall (Blue - Above Center)
    for (int32 i = 0; i < WallAsk.Num(); i++) {
      float Intensity = WallAsk[i];
      if (Intensity > 0.05f) // Threshold
      {
        float ZOffset = (i - 400) * TileHeight;
        FVector Center = ActorLoc + FVector(0, 0, ZOffset);
        FVector Extent(0, TileWidth, TileHeight * 0.45f); // Thin strip

        // Gradient Blue
        FColor Col = FColor::MakeRedToGreenColorFromScalar(
            Intensity); // Debug rainbow? Or pure blue.
        // Let's use Blue with Alpha/Brightness
        uint8 Brightness =
            (uint8)FMath::Clamp(Intensity * 255.0f, 0.0f, 255.0f);
        Col = FColor(0, Brightness / 2, Brightness, 255); // Cyan-Blue

        DrawDebugSolidBox(GetWorld(), Center, Extent, Col, false, -1.0f, 0);
      }
    }

    // Draw Bid Wall (Red - Below Center)
    for (int32 i = 0; i < WallBid.Num(); i++) {
      float Intensity = WallBid[i];
      if (Intensity > 0.05f) {
        float ZOffset = (i - 400) * TileHeight;
        // Note: i-400 is negative for Bids usually?
        // Actually WallBid is same index mapping (-400 to +400).
        // But Bids usually populate the lower half and Asks upper half relative
        // to spot? Let's assume the array index corresponds to Price Tick. So
        // they overlap in Z, but we draw them differently? No, usually Ask >
        // Spot > Bid. So WallAsk will have data at indices > 400. WallBid will
        // have data at indices < 400.

        FVector Center = ActorLoc + FVector(0, 0, ZOffset);
        FVector Extent(0, TileWidth, TileHeight * 0.45f);

        uint8 Brightness =
            (uint8)FMath::Clamp(Intensity * 255.0f, 0.0f, 255.0f);
        FColor Col = FColor(Brightness, 0, 0, 255); // Red

        DrawDebugSolidBox(GetWorld(), Center, Extent, Col, false, -1.0f, 0);
      }
    }
  }
}

void AMwtUdpReceiver::UpdateNiagara() {
  if (!NiagaraComp)
    return;

  FScopeLock Lock(&DataMutex);

  // Push Arrays to User Parameters using Function Library
  UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayFloat(
      NiagaraComp, FName("User.WallAsk"), WallAsk);
  UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayFloat(
      NiagaraComp, FName("User.WallBid"), WallBid);
  UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayFloat(
      NiagaraComp, FName("User.WallErosion"), WallErosion);
  UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayFloat(
      NiagaraComp, FName("User.Vacuum"), Vacuum);
  UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayFloat(
      NiagaraComp, FName("User.PhysicsSigned"), PhysicsSigned);
  UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayFloat(
      NiagaraComp, FName("User.GexAbs"), GexAbs);
  UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayFloat(
      NiagaraComp, FName("User.GexImbalance"), GexImbalance);
}
