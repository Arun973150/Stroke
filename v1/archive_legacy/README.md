# Archive - Legacy Scripts

This folder contains the original Colab-based scripts that were used as the basis for the production-ready pipeline.

## Files

- `global_preprocessing.py` - Original preprocessing script from Google Colab
- `nnu_net_training (1).py` - Original nnU-Net training script from Google Colab

## Note

These scripts have been replaced by the cleaned, production-ready versions in `scripts/`:

| Legacy Script | New Script |
|--------------|------------|
| `global_preprocessing.py` | `scripts/01_preprocess_isles.py` |
| `nnu_net_training (1).py` | `scripts/02_setup_nnunet_dataset.py` + `scripts/03_plan_and_preprocess.py` + `scripts/04_train_nnunet.py` |

The new scripts:
- Remove Colab-specific code (`!` commands, `google.colab` imports)
- Add CLI argument support
- Use YAML configuration
- Add proper logging and monitoring
- Are production-ready for Lambda Labs deployment

## Do Not Use

These legacy scripts are kept for reference only. Use the scripts in `scripts/` folder for training.
