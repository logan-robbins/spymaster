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
  // Create transient textures (Width x Height)
  // Note: UE textures are typically Width x Height, but we store row-major
  // So Width = HistorySeconds (columns), Height = LAYER_HEIGHT (rows)
  
  WallTexture = UTexture2D::CreateTransient(HistorySeconds, LAYER_HEIGHT, PF_B8G8R8A8);
  VacuumTexture = UTexture2D::CreateTransient(HistorySeconds, LAYER_HEIGHT, PF_B8G8R8A8);
  PhysicsTexture = UTexture2D::CreateTransient(HistorySeconds, LAYER_HEIGHT, PF_B8G8R8A8);
  GexTexture = UTexture2D::CreateTransient(HistorySeconds, LAYER_HEIGHT, PF_B8G8R8A8);

  // Configure for pixel-perfect rendering (no filtering, no sRGB)
  auto ConfigureTexture = [](UTexture2D* Tex, const TCHAR* Name) {
    if (!Tex) {
      UE_LOG(LogTemp, Error, TEXT("MwtHeatmapRenderer: Failed to create texture %s"), Name);
      return;
    }
    Tex->Filter = TF_Nearest;
    Tex->SRGB = false;
    Tex->CompressionSettings = TC_VectorDisplacementmap; // No compression
    Tex->MipGenSettings = TMGS_NoMipmaps;
    Tex->AddressX = TA_Wrap; // Ring buffer wrapping
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
  // V1: Using debug drawing instead of materials
  // No material loading needed - all visualization done via DrawDebugSolidBox
  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: Using debug draw for visualization"));
}

void UMwtHeatmapRenderer::CreateMeshes() {
  // Create mesh planes for each layer
  // Layer Z-offsets (in world units, using Y for depth in UE's coordinate system)
  
  // For V1, we skip mesh creation and use debug drawing
  // This is simpler and doesn't require custom materials
  
  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: Skipping mesh creation (using debug draw)"));
}

void UMwtHeatmapRenderer::CreateLayerQuad(UProceduralMeshComponent*& OutMesh, 
                                           float ZOffset, const FString& Name) {
  AActor* Owner = GetOwner();
  if (!Owner) return;

  // Calculate mesh dimensions
  float MeshWidth = HistorySeconds * TimeWorldScale;
  float MeshHeight = LAYER_HEIGHT * TickWorldScale;

  OutMesh = NewObject<UProceduralMeshComponent>(Owner, *Name);
  OutMesh->SetupAttachment(Owner->GetRootComponent());
  OutMesh->RegisterComponent();

  // Create quad vertices
  TArray<FVector> Vertices;
  Vertices.Add(FVector(0, ZOffset, -MeshHeight / 2));           // BL
  Vertices.Add(FVector(MeshWidth, ZOffset, -MeshHeight / 2));   // BR
  Vertices.Add(FVector(MeshWidth, ZOffset, MeshHeight / 2));    // TR
  Vertices.Add(FVector(0, ZOffset, MeshHeight / 2));            // TL

  TArray<int32> Triangles = {0, 2, 1, 0, 3, 2};

  TArray<FVector> Normals;
  TArray<FVector2D> UVs;
  TArray<FColor> VertexColors;
  
  for (int i = 0; i < 4; i++) {
    Normals.Add(FVector(0, -1, 0));
    VertexColors.Add(FColor::White);
  }

  // UVs for texture mapping (0,0 = bottom-left, 1,1 = top-right)
  UVs.Add(FVector2D(0, 1)); // BL
  UVs.Add(FVector2D(1, 1)); // BR
  UVs.Add(FVector2D(1, 0)); // TR
  UVs.Add(FVector2D(0, 0)); // TL

  OutMesh->CreateMeshSection(0, Vertices, Triangles, Normals, UVs,
                              VertexColors, TArray<FProcMeshTangent>(), false);
}

void UMwtHeatmapRenderer::TickComponent(float DeltaTime, ELevelTick TickType,
                                         FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

  // Flush dirty textures to GPU
  if (bWallDirty) {
    UpdateFullTexture(WallTexture, WallData);
    bWallDirty = false;
  }
  if (bVacuumDirty) {
    UpdateFullTexture(VacuumTexture, VacuumData);
    bVacuumDirty = false;
  }
  if (bPhysicsDirty) {
    UpdateFullTexture(PhysicsTexture, PhysicsData);
    bPhysicsDirty = false;
  }
  if (bGexDirty) {
    UpdateFullTexture(GexTexture, GexData);
    bGexDirty = false;
  }

  // Debug draw visualization
  if (!GetWorld()) return;

  AActor* Owner = GetOwner();
  if (!Owner) return;

  FVector Origin = Owner->GetActorLocation();

  // Tile sizes - fill the space (no gaps)
  float TileWidth = TimeWorldScale;   // Full width per second
  float TileHeight = TickWorldScale;  // Full height per tick

  // Visible range
  int32 NumVisibleColumns = FMath::Min(HistorySeconds, 90); // 90 seconds max
  int32 HalfTickRange = VisibleTickRange / 2; // e.g., ±100 ticks from center

  // Debug: count non-empty columns
  static int32 DebugCounter = 0;
  if (++DebugCounter % 60 == 0) {  // Every ~1 second at 60fps
    int32 NonEmptyPhysicsCols = 0;
    for (int32 c = 0; c < NumVisibleColumns; c++) {
      int32 dc = (HeadColumn - c + HistorySeconds) % HistorySeconds;
      for (int32 r = CENTER_IDX - 10; r <= CENTER_IDX + 10; r++) {
        int32 idx = r * HistorySeconds + dc;
        if (PhysicsData.IsValidIndex(idx) && PhysicsData[idx].A > 0) {
          NonEmptyPhysicsCols++;
          break;
        }
      }
    }
    UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: HeadColumn=%d, NonEmptyPhysicsCols=%d of %d"), 
           HeadColumn, NonEmptyPhysicsCols, NumVisibleColumns);
  }
  
  // Draw from newest (right) to oldest (left)
  for (int32 Col = 0; Col < NumVisibleColumns; Col++) {
    int32 DataCol = (HeadColumn - Col + HistorySeconds) % HistorySeconds;
    float XPos = -Col * TimeWorldScale; // Negative X = back in time

    // Only render ticks within VisibleTickRange of center (spot)
    int32 StartRow = CENTER_IDX - HalfTickRange;
    int32 EndRow = CENTER_IDX + HalfTickRange;
    StartRow = FMath::Max(0, StartRow);
    EndRow = FMath::Min(LAYER_HEIGHT - 1, EndRow);

    // Draw Physics layer (back layer - green/red gradient)
    if (bShowPhysics) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        FColor C = PhysicsData[Idx];
        
        if (C.A > 1) {  // Lower threshold to show more data
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileWidth/2, -10, ZPos);
          FVector Extent(TileWidth / 2, 2, TileHeight / 2);
          
          // Physics: G=positive (green/up ease), R=negative (red/down ease)
          FColor DrawColor;
          if (C.G > C.R) {
            DrawColor = FColor(0, FMath::Min(255, (int32)(C.G * 1.5f)), 0, 255);
          } else {
            DrawColor = FColor(FMath::Min(255, (int32)(C.R * 1.5f)), 0, 0, 255);
          }
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }

    // Draw Wall layer (middle layer - blue asks / red bids)
    if (bShowWall) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        FColor C = WallData[Idx];
        
        if (C.A > 5) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileWidth/2, 0, ZPos);
          FVector Extent(TileWidth / 2, 3, TileHeight / 2);
          
          // Wall: B=Ask (cyan/blue above spot), R=Bid (red below spot)
          FColor DrawColor;
          if (C.B > C.R) {
            // Ask (above spot) - cyan/blue
            DrawColor = FColor(0, C.B, FMath::Min(255, (int32)(C.B * 1.2f)), 255);
          } else {
            // Bid (below spot) - red
            DrawColor = FColor(FMath::Min(255, (int32)(C.R * 1.2f)), 0, 0, 255);
          }
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }

    // Draw Vacuum layer (overlay - black/dark)
    if (bShowVacuum) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        FColor C = VacuumData[Idx];
        
        if (C.A > 15) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileWidth/2, 5, ZPos);
          FVector Extent(TileWidth / 2, 1, TileHeight / 2);
          
          // Vacuum: dark overlay
          uint8 Darkness = FMath::Min(200, (int32)(C.A * 1.5f));
          FColor DrawColor(10, 10, 10, Darkness);
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }

    // Draw GEX layer (front layer - green calls / red puts)
    if (bShowGex) {
      for (int32 Row = StartRow; Row <= EndRow; Row++) {
        int32 Idx = GetTextureIndex(Row, DataCol);
        FColor C = GexData[Idx];
        
        if (C.A > 5) {
          float ZPos = (Row - CENTER_IDX) * TickWorldScale;
          FVector Center = Origin + FVector(XPos - TileWidth/2, 10, ZPos);
          FVector Extent(TileWidth / 2, 2, TileHeight / 2);
          
          // GEX: G=calls (green), R=puts (red)
          FColor DrawColor;
          if (C.G > C.R) {
            DrawColor = FColor(0, FMath::Min(255, (int32)(C.G * 1.5f)), 0, 200);
          } else {
            DrawColor = FColor(FMath::Min(255, (int32)(C.R * 1.5f)), 0, 0, 200);
          }
          
          DrawDebugSolidBox(GetWorld(), Center, Extent, DrawColor, false, -1.0f, 0);
        }
      }
    }
  }

  // Draw spot line (at Z=0, the center where spot is)
  if (bShowSpotLine) {
    float LineLength = NumVisibleColumns * TimeWorldScale;
    FVector LineStart = Origin + FVector(0, 15, 0);
    FVector LineEnd = Origin + FVector(-LineLength, 15, 0);
    DrawDebugLine(GetWorld(), LineStart, LineEnd, FColor::Cyan, false, -1.0f, 0, 5.0f);
  }

  // Draw grid lines at $5 intervals (20 ticks)
  if (bShowGridLines) {
    float LineLength = NumVisibleColumns * TimeWorldScale;
    for (int32 Tick = -HalfTickRange; Tick <= HalfTickRange; Tick += 20) {
      float ZPos = Tick * TickWorldScale;
      FVector LineStart = Origin + FVector(0, -15, ZPos);
      FVector LineEnd = Origin + FVector(-LineLength, -15, ZPos);
      
      FColor GridColor;
      float Thickness;
      if (Tick == 0) {
        GridColor = FColor::Yellow;
        Thickness = 3.0f;
      } else {
        GridColor = FColor(80, 80, 80, 255);
        Thickness = 1.0f;
      }
      DrawDebugLine(GetWorld(), LineStart, LineEnd, GridColor, false, -1.0f, 0, Thickness);
    }
    
    // Draw vertical time markers every 10 seconds
    for (int32 Sec = 0; Sec < NumVisibleColumns; Sec += 10) {
      float XPos = -Sec * TimeWorldScale;
      float ZTop = HalfTickRange * TickWorldScale;
      FVector LineStart = Origin + FVector(XPos, -15, -ZTop);
      FVector LineEnd = Origin + FVector(XPos, -15, ZTop);
      DrawDebugLine(GetWorld(), LineStart, LineEnd, FColor(60, 60, 60, 255), false, -1.0f, 0, 1.0f);
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

  // Clear the column first (Wall has τ=0, instant clear)
  ClearColumn(WallData, HeadColumn);

  // Write new data
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

  // Vacuum uses dissipation, so we don't clear - just overwrite
  for (int32 Row = 0; Row < LAYER_HEIGHT; Row++) {
    int32 Idx = GetTextureIndex(Row, HeadColumn);
    float Score = VacuumScores[Row];
    // Frontend: Alpha = vacuum_score * 128
    uint8 Alpha = FMath::Clamp((int32)(Score * 128.0f * VacuumIntensityMult), 0, 255);
    VacuumData[Idx] = FColor(0, 0, 0, Alpha);
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
  
  // Clear new column for wall (instant clear)
  ClearColumn(WallData, HeadColumn);
  bWallDirty = true;
  
  UE_LOG(LogTemp, Log, TEXT("MwtHeatmapRenderer: AdvanceTime %d -> %d"), OldHead, HeadColumn);
}

void UMwtHeatmapRenderer::ApplyDissipation(float DeltaSeconds) {
  // Decay factor: exp(-dt/tau)
  float PhysicsDecay = FMath::Exp(-DeltaSeconds / DECAY_TAU_PHYSICS);
  float VacuumDecay = FMath::Exp(-DeltaSeconds / DECAY_TAU_VACUUM);

  // Apply decay to Physics and Vacuum
  // This is applied to ALL columns for temporal persistence
  
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

// ============== Texture Update Helpers ==============

void UMwtHeatmapRenderer::UpdateFullTexture(UTexture2D* Texture, const TArray<FColor>& Data) {
  if (!Texture || Data.Num() == 0) return;

  // Lock and update texture data
  FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
  void* TextureData = Mip.BulkData.Lock(LOCK_READ_WRITE);
  
  if (TextureData) {
    FMemory::Memcpy(TextureData, Data.GetData(), Data.Num() * sizeof(FColor));
    Mip.BulkData.Unlock();
    Texture->UpdateResource();
  }
}

void UMwtHeatmapRenderer::ClearColumn(TArray<FColor>& Data, int32 Column) {
  for (int32 Row = 0; Row < LAYER_HEIGHT; Row++) {
    int32 Idx = GetTextureIndex(Row, Column);
    if (Idx < Data.Num()) {
      Data[Idx] = FColor(0, 0, 0, 0);
    }
  }
}

// ============== Utility Methods ==============

int32 UMwtHeatmapRenderer::GetTextureIndex(int32 Row, int32 Column) const {
  // Row-major layout: Texture is Width (HistorySeconds) x Height (LAYER_HEIGHT)
  // Index = Row * Width + Column
  return Row * HistorySeconds + Column;
}

FColor UMwtHeatmapRenderer::IntensityToWallColor(float AskIntensity, float BidIntensity) const {
  // Wall: R=Bid intensity, B=Ask intensity, A=max for visibility
  uint8 Ask = FMath::Clamp((int32)(AskIntensity * 255.0f * WallIntensityMult), 0, 255);
  uint8 Bid = FMath::Clamp((int32)(BidIntensity * 255.0f * WallIntensityMult), 0, 255);
  uint8 Alpha = FMath::Max(Ask, Bid);
  return FColor(Bid, 0, Ask, Alpha);
}

FColor UMwtHeatmapRenderer::ScoreToPhysicsColor(float SignedScore) const {
  // Physics: +ve = green (up ease), -ve = red (down ease)
  float Scaled = SignedScore * PhysicsIntensityMult;
  uint8 Intensity = FMath::Clamp((int32)(FMath::Abs(Scaled) * 255.0f), 0, 255);
  
  if (Scaled > 0.01f) {
    return FColor(0, Intensity, 0, Intensity); // Green
  } else if (Scaled < -0.01f) {
    return FColor(Intensity, 0, 0, Intensity); // Red
  }
  return FColor(0, 0, 0, 0);
}

FColor UMwtHeatmapRenderer::GexToColor(float Abs, float Imbalance) const {
  // GEX: Imbalance +ve = calls (green), -ve = puts (red)
  uint8 Intensity = FMath::Clamp((int32)(Abs * 255.0f * GexIntensityMult), 0, 255);
  
  if (Intensity < 5) return FColor(0, 0, 0, 0);
  
  if (Imbalance > 0.1f) {
    return FColor(0, Intensity, 0, Intensity); // Green (calls)
  } else if (Imbalance < -0.1f) {
    return FColor(Intensity, 0, 0, Intensity); // Red (puts)
  }
  return FColor(Intensity, Intensity, 0, Intensity); // Yellow (balanced)
}
