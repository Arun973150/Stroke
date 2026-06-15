import json
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent

def plot_history(name, folder):
    hist_path = ROOT / folder / "training_history.json"
    if not hist_path.exists():
        print(f"Missing: {hist_path}")
        return
    with hist_path.open() as f:
        hist = json.load(f)
    if not hist:
        print(f"Empty history: {hist_path}")
        return

    epochs = [h.get("epoch", i+1) for i, h in enumerate(hist)]
    val_dice = [h.get("val_dice", None) for h in hist]
    strat = [h.get("stratified", {}) for h in hist]
    tiny   = [s.get("tiny")   for s in strat]
    small  = [s.get("small")  for s in strat]
    medium = [s.get("medium") for s in strat]
    large  = [s.get("large")  for s in strat]

    plt.figure(figsize=(8,5))
    plt.plot(epochs, val_dice, label="Val Dice")
    if any(t is not None for t in tiny):   plt.plot(epochs, tiny,   label="Tiny",   alpha=0.6)
    if any(s is not None for s in small):  plt.plot(epochs, small,  label="Small",  alpha=0.6)
    if any(m is not None for m in medium): plt.plot(epochs, medium, label="Medium", alpha=0.6)
    if any(l is not None for l in large):  plt.plot(epochs, large,  label="Large",  alpha=0.6)
    plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.title(f"Training: {name}")
    plt.legend(); plt.grid(True); plt.tight_layout()

    out = ROOT / "results" / "visualizations" / f"training_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

if __name__ == "__main__":
    plot_history("v6", "ablation_se_attention")
    plot_history("v8_tiny", "fusion_results_v8_tiny")