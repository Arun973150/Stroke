"""
ISLES-2022 Winner-Style Custom nnU-Net Trainer
================================================
Key change: Dice + TopK CE loss (focuses on hardest 10% voxels)
This is the single highest-impact change for small lesion sensitivity.

Setup:
  # Copy to nnU-Net's trainer variants directory:
  cp scripts/nnunet_custom_trainer.py \
     ~/sentinel_v2/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainerISLES.py

  # Train:
  nnUNetv2_train 100 3d_fullres 2 -tr nnUNetTrainerISLES
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_topk_loss


class nnUNetTrainerISLES(nnUNetTrainer):
    """Dice + TopK CE loss, 1500 epochs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 1500

    def _build_loss(self):
        return DC_and_topk_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            log_dice=False,
            ignore_label=self.label_manager.ignore_label,
        )
