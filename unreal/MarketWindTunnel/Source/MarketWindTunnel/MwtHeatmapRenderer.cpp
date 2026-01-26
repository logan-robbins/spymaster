#include "MwtHeatmapRenderer.h"
#include "Engine/Texture2D.h"
#include "DrawDebugHelpers.h"

UMwtHeatmapRenderer::UMwtHeatmapRenderer() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.TickGroup = TG_PostUpdateWork;
}

void UMwtHeatmapRenderer::BeginPlay() {
  Super::BeginPlay();

  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: Initializing with %d history seconds, %d layer height"),
         HistorySeconds, LAYER_HEIGHT);

  InitializeTextureData();
  CreateTextures();
  CreateMaterials();
  CreateMeshes();

  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: Initialization complete"));
}

void UMwtHeatmapRenderer::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  Super::EndPlay(EndPlayReason);
}

void UMwtHeatmapRenderer::InitializeTextureData() {
  int32 TotalPixels = HistorySeconds * LAYER_HEIGHT;

  // Initialize all arrays with transparent black
  WallData.Init(FColor(0, 0, 0, 0), TotalPixels);
  VacuumData.Init(FColor(0, 0, 0, 0), TotalPixels);
  PhysicsData.Init(FColor(0, 0, 0, 0), TotalPixels);
  GexData.Init(FColor(0, 0, 0, 0), TotalPixels);
  SpotHistory.Init(0.0f, HistorySeconds);

  HeadColumn = 0;
  CurrentSpotRefTick = 0.0f;

  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: Allocated %d pixels per texture"), TotalPixels);
}

void UMwtHeatmapRenderer::CreateTextures() {
  WallTexture = UTexture2D::CreateTransient(HistorySeconds, LAYER_HEIGHT, PF_B8G8R8A8);
  VacuumTexture = UTexture2D::CreateTransient(HistorySeconds, LAYER_HEIGHT, PF_B8G8R8A8);
  PhysicsTexture = UTexture2D::CreateTransient(HistorySeconds, LAYER_HEIGHT, PF_B8G8R8A8);
  GexTexture = UTexture2D::CreateTransient(HistorySeconds, LAYER_HEIGHT, PF_B8G8R8A8);

  auto ConfigureTexture = [](UTexture2D* Tex, const TCHAR* Name) {
    if (!Tex) {
      UE_LOG(LogTemp, Error, TEXT("MwtHeatmapRenderer: Failed to create texture %s"), Name);
      return;
    }
    Tex->Filter = TF_Nearest;
    Tex->SRGB = false;
    Tex->CompressionSettings = TC_VectorDisplacementmap;
    Tex->MipGenSettings = TMGS_NoMipmaps;
    Tex->AddressX = TA_Wrap;
    Tex->AddressY = TA_Clamp;
    Tex->UpdateResource();
    UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: Created texture %s"), Name);
  };

  ConfigureTexture(WallTexture, TEXT("Wall"));
  ConfigureTexture(VacuumTexture, TEXT("Vacuum"));
  ConfigureTexture(PhysicsTexture, TEXT("Physics"));
  ConfigureTexture(GexTexture, TEXT("GEX"));
}

void UMwtHeatmapRenderer::CreateMaterials() {
  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: Using debug draw for visualization"));
}

void UMwtHeatmapRenderer::CreateMeshes() {
  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: Skipping mesh creation (using debug draw)"));
}

void UMwtHeatmapRenderer::TickComponent(float DeltaTime, ELevelTick TickType,
                                         FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

  if (!GetWorld()) return;
  AActor* Owner = GetOwner();
  if (!Owner) return;

  FVector Origin = Owner->GetActorLocation();

  // Tile dimensions
  float TileW = TimeWorldScale;
  float TileH = TickWorldScale;

  // Visible range  
  int32 NumCols = FMath::Min(HistorySeconds, 90);
  int32 HalfTicks = VisibleTickRange / 2;

  // Color scheme from UPDATE.md Section 7.1:
  // - High resistance/blocked → RED (warm) 
  // - Vacuum/cavitation → CYAN/BLUE (cool)
  // - Physics ease: easier path = brighter

  for (int32 Col = 0; Col < NumCols; Col++) {
    int32 DataCol = (HeadColumn - Col + HistorySeconds) % HistorySeconds;
    float XPos = -Col * TimeWorldScale;

    int32 StartRow = CENTER_IDX - HalfTicks;
    int32 EndRow = CENTER_IDX + HalfTicks;
    StartRow = FMath::Max(0, StartRow);
    EndRow = FMath::Min(LAYER_HEIGHT - 1, EndRow);

    // === LAYER 1: PHYSICS (back, Z=-500) ===
    // Shows directional ease - where price can flow
    // Brighter = easier to traverse
    if (bShowPhysics) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        if (!PhysicsData.IsValidIndex(Idx)) continue;
        FColor C = PhysicsData[Idx];
        
        if (C.A > 0) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileW/2, -500, ZPos);
          FVector Extent(TileW/2, 5, TileH/2);
          
          // Physics: G channel = up ease, R channel = down ease
          // Use green for up ease (above spot), red for down ease (below spot)
          FColor DrawColor;
          if (Row > CENTER_IDX) {
            // Above spot: green = easy to move up through
            DrawColor = FColor(0, C.G, C.G/3, 255);
          } else {
            // Below spot: red = easy to move down through  
            DrawColor = FColor(C.R, 0, 0, 255);
          }
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }

    // === LAYER 2: WALL (middle, Z=0) ===
    // Liquidity/resistance - RED = blocked, intensity = how solid
    if (bShowWall) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        if (!WallData.IsValidIndex(Idx)) continue;
        FColor C = WallData[Idx];
        
        if (C.A > 0) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileW/2, 0, ZPos);
          FVector Extent(TileW/2, 10, TileH/2);
          
          // Wall: R=bid intensity, B=ask intensity
          // Both are "resistance" - render as warm colors (red/orange)
          uint8 MaxIntensity = FMath::Max(C.R, C.B);
          // Warm color: more red, some orange tint
          FColor DrawColor(MaxIntensity, MaxIntensity/3, 0, 255);
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }

    // === LAYER 3: VACUUM (front overlay, Z=+200) ===
    // Cavitation/erosion - CYAN = vacuum forming
    if (bShowVacuum) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        if (!VacuumData.IsValidIndex(Idx)) continue;
        FColor C = VacuumData[Idx];
        
        if (C.A > 0) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileW/2, 200, ZPos);
          FVector Extent(TileW/2, 3, TileH/2);
          
          // Vacuum: CYAN (low pressure / cavitation)
          FColor DrawColor(0, C.A, C.A, 200);
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }

    // === LAYER 4: GEX (front, Z=+400) ===
    // Options gamma - stiffness ridges
    if (bShowGex) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        if (!GexData.IsValidIndex(Idx)) continue;
        FColor C = GexData[Idx];
        
        if (C.A > 0) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileW/2, 400, ZPos);
          FVector Extent(TileW/2, 5, TileH/2);
          
          // GEX: Purple/magenta for stiffness
          FColor DrawColor(C.A, 0, C.A, 200);
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }
  }

  // === SPOT LINE (frontmost, bright cyan) ===
  if (bShowSpotLine) {
    float LineLen = NumCols * TimeWorldScale;
    FVector Start = Origin + FVector(0, 500, 0);
    FVector End = Origin + FVector(-LineLen, 500, 0);
    DrawDebugLine(GetWorld(), Start, End, FColor::Cyan, false, -1.0f, 0, 8.0f);
  }

  // === GRID LINES ===
  if (bShowGridLines) {
    float LineLen = NumCols * TimeWorldScale;
    
    // Horizontal price lines every $5 (20 ticks)
    for (int32 Tick = -HalfTicks; Tick <= HalfTicks; Tick += 20) {
      float ZPos = Tick * TickWorldScale;
      FVector Start = Origin + FVector(0, -600, ZPos);
      FVector End = Origin + FVector(-LineLen, -600, ZPos);
      
      FColor GridColor = (Tick == 0) ? FColor::Yellow : FColor(60, 60, 60, 255);
      float Thick = (Tick == 0) ? 4.0f : 1.0f;
      DrawDebugLine(GetWorld(), Start, End, GridColor, false, -1.0f, 0, Thick);
    }
    
    // Vertical time lines every 10 seconds
    for (int32 Sec = 0; Sec < NumCols; Sec += 10) {
      float XPos = -Sec * TimeWorldScale;
      float ZTop = HalfTicks * TickWorldScale;
      FVector Start = Origin + FVector(XPos, -600, -ZTop);
      FVector End = Origin + FVector(XPos, -600, ZTop);
      DrawDebugLine(GetWorld(), Start, End, FColor(60, 60, 60, 255), false, -1.0f, 0, 1.0f);
    }
  }
}

// ============== Data Update Methods ==============

void UMwtHeatmapRenderer::UpdateWallColumn(const TArray<float>& AskIntensities,
                                            const TArray<float>& BidIntensities) {
  if (AskIntensities.Num() != LAYER_HEIGHT || BidIntensities.Num() != LAYER_HEIGHT) {
    UE_LOG(LogTemp, Warning, TEXT("MwtHeatmapRenderer: Wall data size mismatch (%d, %d) expected %d"),
           AskIntensities.Num(), BidIntensities.Num(), LAYER_HEIGHT);
    return;
  }

  for (int32 Row = 0; Row < LAYER_HEIGHT; Row++) {
    int32 Idx = GetTextureIndex(Row, HeadColumn);
    WallData[Idx] = IntensityToWallColor(AskIntensities[Row], BidIntensities[Row]);
  }

  bWallDirty = true;
}

void UMwtHeatmapRenderer::UpdateVacuumColumn(const TArray<float>& VacuumScores) {
  if (VacuumScores.Num() != LAYER_HEIGHT) {
    UE_LOG(LogTemp, Warning, TEXT("MwtHeatmapRenderer: Vacuum data size mismatch"));
    return;
  }

  for (int32 Row = 0; Row < LAYER_HEIGHT; Row++) {
    int32 Idx = GetTextureIndex(Row, HeadColumn);
    float Score = VacuumScores[Row];
    uint8 Alpha = FMath::Clamp((int32)(Score * 255.0f * VacuumIntensityMult), 0, 255);
    VacuumData[Idx] = FColor(0, Alpha, Alpha, Alpha);  // Cyan tint
  }

  bVacuumDirty = true;
}

void UMwtHeatmapRenderer::UpdatePhysicsColumn(const TArray<float>& PhysicsSignedScores) {
  if (PhysicsSignedScores.Num() != LAYER_HEIGHT) {
    UE_LOG(LogTemp, Warning, TEXT("MwtHeatmapRenderer: Physics data size mismatch"));
    return;
  }

  for (int32 Row = 0; Row < LAYER_HEIGHT; Row++) {
    int32 Idx = GetTextureIndex(Row, HeadColumn);
    PhysicsData[Idx] = ScoreToPhysicsColor(PhysicsSignedScores[Row]);
  }

  bPhysicsDirty = true;
}

void UMwtHeatmapRenderer::UpdateGexColumn(const TArray<float>& GexAbs,
                                           const TArray<float>& GexImbalance) {
  if (GexAbs.Num() != LAYER_HEIGHT || GexImbalance.Num() != LAYER_HEIGHT) {
    UE_LOG(LogTemp, Warning, TEXT("MwtHeatmapRenderer: GEX data size mismatch"));
    return;
  }

  for (int32 Row = 0; Row < LAYER_HEIGHT; Row++) {
    int32 Idx = GetTextureIndex(Row, HeadColumn);
    GexData[Idx] = GexToColor(GexAbs[Row], GexImbalance[Row]);
  }

  bGexDirty = true;
}

void UMwtHeatmapRenderer::UpdateSpotRef(float SpotRefTick) {
  CurrentSpotRefTick = SpotRefTick;
  if (HeadColumn < SpotHistory.Num()) {
    SpotHistory[HeadColumn] = SpotRefTick;
  }
}

void UMwtHeatmapRenderer::AdvanceTime() {
  int32 OldHead = HeadColumn;
  HeadColumn = (HeadColumn + 1) % HistorySeconds;
  
  // Clear ONLY wall for new column (Wall has τ=0)
  // Physics and Vacuum persist (τ=5s decay handled by receiver)
  ClearColumn(WallData, HeadColumn);
  
  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: AdvanceTime %d -> %d"), OldHead, HeadColumn);
}

void UMwtHeatmapRenderer::ApplyDissipation(float DeltaSeconds) {
  float PhysicsDecay = FMath::Exp(-DeltaSeconds / DECAY_TAU_PHYSICS);
  float VacuumDecay = FMath::Exp(-DeltaSeconds / DECAY_TAU_VACUUM);

  for (int32 i = 0; i < PhysicsData.Num(); i++) {
    FColor& C = PhysicsData[i];
    C.R = (uint8)(C.R * PhysicsDecay);
    C.G = (uint8)(C.G * PhysicsDecay);
    C.A = (uint8)(C.A * PhysicsDecay);
  }

  for (int32 i = 0; i < VacuumData.Num(); i++) {
    FColor& C = VacuumData[i];
    C.A = (uint8)(C.A * VacuumDecay);
  }

  bPhysicsDirty = true;
  bVacuumDirty = true;
}

void UMwtHeatmapRenderer::FlushTextures() {
  UpdateFullTexture(WallTexture, WallData);
  UpdateFullTexture(VacuumTexture, VacuumData);
  UpdateFullTexture(PhysicsTexture, PhysicsData);
  UpdateFullTexture(GexTexture, GexData);
  
  bWallDirty = false;
  bVacuumDirty = false;
  bPhysicsDirty = false;
  bGexDirty = false;
}

// ============== Helpers ==============

void UMwtHeatmapRenderer::UpdateFullTexture(UTexture2D* Texture, const TArray<FColor>& Data) {
  if (!Texture || Data.Num() == 0) return;

  FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
  void* TextureData = Mip.BulkData.Lock(LOCK_READ_WRITE);
  FMemory::Memcpy(TextureData, Data.GetData(), Data.Num() * sizeof(FColor));
  Mip.BulkData.Unlock();
  Texture->UpdateResource();
}

void UMwtHeatmapRenderer::ClearColumn(TArray<FColor>& Data, int32 Column) {
  for (int32 Row = 0; Row < LAYER_HEIGHT; Row++) {
    int32 Idx = GetTextureIndex(Row, Column);
    if (Data.IsValidIndex(Idx)) {
      Data[Idx] = FColor(0, 0, 0, 0);
    }
  }
}

int32 UMwtHeatmapRenderer::GetTextureIndex(int32 Row, int32 Col) const {
  return Row * HistorySeconds + Col;
}

FColor UMwtHeatmapRenderer::IntensityToWallColor(float AskIntensity, float BidIntensity) const {
  // Wall = resistance/mass = warm colors (red/orange)
  uint8 Ask = FMath::Clamp((int32)(AskIntensity * 255.0f * WallIntensityMult), 0, 255);
  uint8 Bid = FMath::Clamp((int32)(BidIntensity * 255.0f * WallIntensityMult), 0, 255);
  uint8 Alpha = FMath::Max(Ask, Bid);
  // Store both so we can distinguish asks vs bids if needed
  return FColor(Bid, 0, Ask, Alpha);
}

FColor UMwtHeatmapRenderer::ScoreToPhysicsColor(float SignedScore) const {
  // Physics = ease of movement
  // Positive (up ease) stored in G, Negative (down ease) stored in R
  float Scaled = SignedScore * PhysicsIntensityMult;
  uint8 Intensity = FMath::Clamp((int32)(FMath::Abs(Scaled) * 255.0f), 0, 255);
  
  if (Scaled > 0.001f) {
    return FColor(0, Intensity, 0, Intensity);  // Green = up ease
  } else if (Scaled < -0.001f) {
    return FColor(Intensity, 0, 0, Intensity);  // Red = down ease
  }
  return FColor(0, 0, 0, 0);
}

FColor UMwtHeatmapRenderer::GexToColor(float GexAbsVal, float ImbalanceRatio) const {
  // GEX = stiffness from options
  uint8 Intensity = FMath::Clamp((int32)(GexAbsVal * 255.0f * GexIntensityMult), 0, 255);
  // Purple/magenta for stiffness
  return FColor(Intensity, 0, Intensity, Intensity);
}

void UMwtHeatmapRenderer::CreateQuadMesh(UProceduralMeshComponent*& OutMesh, 
                                          float MeshWidth, float MeshHeight, 
                                          float ZOffset, const FString& Name) {
  // Not used for debug draw
}
