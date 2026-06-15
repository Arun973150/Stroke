#!/usr/bin/env python3
"""
Verify the integrity of downloaded model checkpoints.
"""
import torch
import os
from pathlib import Path

def verify_checkpoint(path):
    """Attempt to load a checkpoint and check for basic structure."""
    path = Path(path)
    if not path.exists():
        return False, "File does not exist"
    
    try:
        # Load on CPU for verification
        # Using weights_only=False because these are complex dictionaries
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Check basic keys
        keys = checkpoint.keys()
        size_mb = path.stat().st_size / (1024 * 1024)
        
        return True, f"OK ({size_mb:.1f} MB, Keys: {list(keys)[:3]}...)"
    except Exception as e:
        return False, f"CORRUPT: {str(e)}"

def main():
    base_dir = Path("trained_models")
    models = {
        "SegResNet": base_dir / "segresnet",
        "Swin-UNETR": base_dir / "swin_unetr",
        "nnU-Net": base_dir / "nnunet"
    }
    
    print("=" * 60)
    print("Checkpoint Integrity Verification")
    print("=" * 60)
    
    for model_name, model_dir in models.items():
        print(f"\n[bold]{model_name}[/bold]")
        if not model_dir.exists():
            print(f"  Directory not found: {model_dir}")
            continue
            
        # Look for fold directories
        folds = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
        
        if not folds:
            print("  No fold directories found.")
            continue
            
        for fold in folds:
            # Check for common checkpoint names
            found = False
            for target in ["checkpoint_best.pth", "checkpoint_final.pth"]:
                ckpt_path = fold / target
                if ckpt_path.exists():
                    success, msg = verify_checkpoint(ckpt_path)
                    status = "[PASS]" if success else "[FAIL]"
                    print(f"  {status} {fold.name}/{target}: {msg}")
                    found = True
                    break
            
            if not found:
                print(f"  [MISSING] {fold.name}: No .pth file found")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
