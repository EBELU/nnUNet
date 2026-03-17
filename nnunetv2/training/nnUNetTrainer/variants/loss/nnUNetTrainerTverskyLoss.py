import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.tversky_focal import MemoryEfficientSoftTverskyDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerTverskyLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = MemoryEfficientSoftTverskyDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': self.label_manager.has_regions, 'smooth': 1e-5, 'ddp': self.is_ddp},
                            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerTverskyCELoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.alpha = 0.7
        self.beta = 0.3
        self.gamma = 1

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp, "alpha": self.alpha, "beta": self.beta , "focal_gamma": self.gamma},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftTverskyDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp, "alpha": self.alpha, "beta": self.beta , "focal_gamma": self.gamma}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftTverskyDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
    
class nnUNetTrainerTverskyCELoss_a07b03g1(nnUNetTrainerTverskyCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        
class nnUNetTrainerTverskyDiceCELoss_a03b07g1_250(nnUNetTrainerTverskyCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.alpha = 0.3
        self.beta = 0.7

class nnUNetTrainerTverskyDiceCELoss_a03b07g1_750(nnUNetTrainerTverskyCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 750
        self.alpha = 0.3
        self.beta = 0.7

class nnUNetTrainerTverskyCELoss_a02b08g075(nnUNetTrainerTverskyCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.alpha = 0.2
        self.beta = 0.8
        self.gamma = 0.75

class nnUNetTrainerTverskyCELoss_a01b09g1(nnUNetTrainerTverskyCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.alpha = 0.1
        self.beta = 0.9
        self.gamma = 1

class nnUNetTrainerTverskyCELoss_a005b095g075(nnUNetTrainerTverskyCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.alpha = 0.05
        self.beta = 0.95
        self.gamma = 0.75