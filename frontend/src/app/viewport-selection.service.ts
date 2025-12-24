import { Injectable, computed, signal, WritableSignal, Signal } from '@angular/core';
import { DataStreamService, ViewportTarget } from './data-stream.service';
import { DerivedLevel } from './level-derived.service';

/**
 * Viewport Selection Service
 * 
 * Purpose: Manage trader's focus on specific levels.
 * 
 * Philosophy:
 * - Viewport = trader's "radar" showing relevant levels
 * - Selected level = current focus for detailed analysis
 * - Can pin a level to keep focus even as it moves out of band
 * 
 * For 0DTE traders:
 * - Default: Show highest utility level (ML recommends this)
 * - Manual: Click to select specific level
 * - Pinned: Lock focus until manually released
 */
@Injectable({
    providedIn: 'root'
})
export class ViewportSelectionService {
    private dataStream = inject(DataStreamService);
    
    /**
     * Currently selected level ID (null = auto-select highest utility)
     */
    private selectedLevelId: WritableSignal<string | null> = signal(null);
    
    /**
     * Whether current selection is pinned (won't auto-update)
     */
    public isPinned: WritableSignal<boolean> = signal(false);
    
    /**
     * Available viewport targets sorted by utility
     */
    public viewportTargets: Signal<ViewportTarget[]> = computed(() => {
        const viewport = this.dataStream.viewportData();
        if (!viewport) return [];
        
        // Sort by utility score descending
        return [...viewport.targets].sort((a, b) => b.utility_score - a.utility_score);
    });
    
    /**
     * Currently focused viewport target
     * 
     * Logic:
     * - If pinned and level exists: return pinned level
     * - If manual selection exists: return that level
     * - Otherwise: return highest utility level
     */
    public selectedTarget: Signal<ViewportTarget | null> = computed(() => {
        const targets = this.viewportTargets();
        if (targets.length === 0) return null;
        
        const selectedId = this.selectedLevelId();
        
        // Find selected level if exists
        if (selectedId) {
            const found = targets.find(t => t.level_id === selectedId);
            if (found) return found;
            
            // Selected level no longer in viewport - clear selection unless pinned
            if (!this.isPinned()) {
                this.selectedLevelId.set(null);
            }
        }
        
        // Default: highest utility level
        return targets[0] || null;
    });
    
    /**
     * Select a specific level by ID
     */
    public selectLevel(levelId: string, pin: boolean = false): void {
        this.selectedLevelId.set(levelId);
        this.isPinned.set(pin);
    }
    
    /**
     * Clear selection (return to auto-select)
     */
    public clearSelection(): void {
        this.selectedLevelId.set(null);
        this.isPinned.set(false);
    }
    
    /**
     * Toggle pin state
     */
    public togglePin(): void {
        this.isPinned.update(p => !p);
    }
    
    /**
     * Check if a level is currently selected
     */
    public isSelected(levelId: string): boolean {
        const target = this.selectedTarget();
        return target?.level_id === levelId;
    }
    
    /**
     * Get selection mode description for UI
     */
    public getSelectionMode(): Signal<string> {
        return computed(() => {
            const isPinned = this.isPinned();
            const hasSelection = this.selectedLevelId() !== null;
            
            if (isPinned && hasSelection) {
                return 'Pinned';
            } else if (hasSelection) {
                return 'Manual';
            } else {
                return 'Auto (Highest Utility)';
            }
        });
    }
}

// Import inject
import { inject } from '@angular/core';

