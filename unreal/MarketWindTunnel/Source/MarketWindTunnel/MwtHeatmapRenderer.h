#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ProceduralMeshComponent.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "MwtHeatmapRenderer.generated.h"

/**
 * MwtHeatmapRenderer - Tick-Native Market Visualization
 * 
 * Replicates the frontend WebGL renderer in Unreal Engine.
 * Architecture matches DOCS_FRONTEND.md exactly:
 * 
 * Coordinate Systems:
 *   - Data Space: Integer ticks, spot_ref_price_int anchor
 *   - Texture Space: Y=0..801 (layer height), X=0..HistorySeconds (time)
 *   - World Space: 1 tick = TickWorldScale units, 1 second = TimeWorldScale units
 * 
 * Layer Stack (Z-order back→front):
 *   Z=-0.02: Physics (green above / red below)
 *   Z= 0.00: Wall (blue asks / red bids)
 *   Z= 0.01: GEX (green calls / red puts)
 *   Z= 0.015: Vacuum (black overlay)
 *   Z= 1.00: Spot Line (cyan)
 * 
 * Dissipation Model:
 *   Physics: τ=5s (decays ~18% per second)
 *   Vacuum: τ=5s
 *   Wall: τ=0 (instant clear)
 *   GEX: preserved
 */
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class MARKETWINDTUNNEL_API UMwtHeatmapRenderer : public UActorComponent {
  GENERATED_BODY()

public:
  UMwtHeatmapRenderer();

  // ============== Configuration ==============
  
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Scale")
  float TickWorldScale = 50.0f; // 50 units per tick = MUCH larger

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Scale")
  float TimeWorldScale = 100.0f; // 100 units per second = MUCH wider

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Scale")
  int32 HistorySeconds = 120; // 2 minutes of visible history
  
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Scale")
  int32 VisibleTickRange = 160; // ±80 ticks ($20 range) - focused view

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Intensity")
  float WallIntensityMult = 10.0f;  // Boosted

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Intensity")
  float VacuumIntensityMult = 5.0f;  // Boosted

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Intensity")
  float PhysicsIntensityMult = 20.0f;  // Physics needs big boost

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Intensity")
  float GexIntensityMult = 5.0f;  // Boosted

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Layers")
  bool bShowWall = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Layers")
  bool bShowVacuum = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Layers")
  bool bShowPhysics = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Layers")
  bool bShowGex = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Layers")
  bool bShowSpotLine = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MWT|Layers")
  bool bShowGridLines = true;

  // ============== Data Update Interface ==============
  
  /** Update wall layer data for current time column */
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void UpdateWallColumn(const TArray<float>& AskIntensities, 
                        const TArray<float>& BidIntensities);

  /** Update vacuum layer data for current time column */
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void UpdateVacuumColumn(const TArray<float>& VacuumScores);

  /** Update physics layer data for current time column */
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void UpdatePhysicsColumn(const TArray<float>& PhysicsSignedScores);

  /** Update GEX layer data for current time column */
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void UpdateGexColumn(const TArray<float>& GexAbs, 
                       const TArray<float>& GexImbalance);

  /** Update spot reference tick (used for spot line and history) */
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void UpdateSpotRef(float SpotRefTick);

  /** Advance ring buffer to next time column (call once per second) */
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void AdvanceTime();

  /** Apply temporal dissipation to all layers */
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void ApplyDissipation(float DeltaSeconds);

  /** Get current head column index */
  UFUNCTION(BlueprintPure, Category = "MWT")
  int32 GetHeadColumn() const { return HeadColumn; }

  /** Force full texture update (call after bulk data changes) */
  UFUNCTION(BlueprintCallable, Category = "MWT")
  void FlushTextures();

protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
  virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                             FActorComponentTickFunction* ThisTickFunction) override;

private:
  // ============== Constants ==============
  static constexpr int32 LAYER_HEIGHT = 801;  // ±400 ticks
  static constexpr int32 CENTER_IDX = 400;    // rel_ticks=0 maps to index 400
  static constexpr float DECAY_TAU_PHYSICS = 5.0f;  // seconds
  static constexpr float DECAY_TAU_VACUUM = 5.0f;   // seconds

  // ============== Ring Buffer State ==============
  int32 HeadColumn = 0;
  float CurrentSpotRefTick = 0.0f;

  // ============== CPU-Side Texture Data ==============
  // Format: Row-major, [Row * Width + Column]
  // Each pixel: RGBA where channels encode layer-specific data
  TArray<FColor> WallData;      // R=Bid, B=Ask, A=max(Bid,Ask)
  TArray<FColor> VacuumData;    // R=G=B=0, A=vacuum_score*255
  TArray<FColor> PhysicsData;   // G=positive, R=negative, A=abs
  TArray<FColor> GexData;       // G=calls, R=puts, A=abs
  TArray<float> SpotHistory;    // Spot ref tick per column (for rectification)

  // ============== GPU Resources ==============
  UPROPERTY()
  UTexture2D* WallTexture;
  UPROPERTY()
  UTexture2D* VacuumTexture;
  UPROPERTY()
  UTexture2D* PhysicsTexture;
  UPROPERTY()
  UTexture2D* GexTexture;

  UPROPERTY()
  UMaterialInstanceDynamic* WallMaterial;
  UPROPERTY()
  UMaterialInstanceDynamic* VacuumMaterial;
  UPROPERTY()
  UMaterialInstanceDynamic* PhysicsMaterial;
  UPROPERTY()
  UMaterialInstanceDynamic* GexMaterial;
  UPROPERTY()
  UMaterialInstanceDynamic* SpotLineMaterial;

  UPROPERTY()
  UProceduralMeshComponent* WallMesh;
  UPROPERTY()
  UProceduralMeshComponent* VacuumMesh;
  UPROPERTY()
  UProceduralMeshComponent* PhysicsMesh;
  UPROPERTY()
  UProceduralMeshComponent* GexMesh;
  UPROPERTY()
  UProceduralMeshComponent* SpotLineMesh;
  UPROPERTY()
  UProceduralMeshComponent* GridLinesMesh;

  // ============== Base Materials (loaded from content) ==============
  UPROPERTY()
  UMaterial* BaseMaterial;

  // ============== Initialization ==============
  void InitializeTextureData();
  void CreateTextures();
  void CreateMaterials();
  void CreateMeshes();
  void CreateLayerQuad(UProceduralMeshComponent*& OutMesh, float ZOffset, 
                       const FString& Name);
  void CreateSpotLineMesh();
  void CreateGridLinesMesh();

  // ============== Texture Updates ==============
  void UpdateTextureColumn(TArray<FColor>& Data, UTexture2D* Texture, 
                           int32 Column, const TArray<FColor>& ColumnPixels);
  void UpdateFullTexture(UTexture2D* Texture, const TArray<FColor>& Data);
  void ClearColumn(TArray<FColor>& Data, int32 Column);

  // ============== Material Updates ==============
  void UpdateMaterialParameters();

  // ============== Utility ==============
  int32 GetTextureIndex(int32 Row, int32 Column) const;
  FColor IntensityToWallColor(float AskIntensity, float BidIntensity) const;
  FColor ScoreToPhysicsColor(float SignedScore) const;
  FColor GexToColor(float Abs, float Imbalance) const;

  // Track if textures need GPU update
  bool bWallDirty = false;
  bool bVacuumDirty = false;
  bool bPhysicsDirty = false;
  bool bGexDirty = false;
};
