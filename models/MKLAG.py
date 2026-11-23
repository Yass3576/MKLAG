import torch
import pytorch_lightning as pl
from einops import rearrange

from models.blocksLag import mklag
from models.optim import build_optimizer, build_scheduler

LAG_KEYS = {
    's1': 'lag_s1', 'e1': 'lag_e1',
    's2': 'lag_s2', 'e2': 'lag_e2',
    's3': 'lag_s3', 'e3': 'lag_e3',
    'n_segment': 'n_segment',
}

GT_LAG_FIELDS = ['has_seg1', 'lag_s1', 'lag_e1',
                 'has_seg2', 'lag_s2', 'lag_e2',
                 'has_seg3', 'lag_s3', 'lag_e3']


class MKLag(pl.LightningModule):
    """
    @brief Lightning wrapper for MKLag with morphokinetic, developmental stage and Lag phase prediction.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg
        self.n_actions = cfg.n_actions
        self.n_tclasses = cfg.n_tclasses

        # Core model
        self.model = mklag(cfg, cfg.in_dim, cfg.n_actions, cfg.n_tclasses)

        # Optimisation
        self.optimizer_name = cfg.opti.optimizer
        self.lr = cfg.opti.learning_rate
        self.l2 = cfg.opti.l2
        self.schedule = cfg.opti.schedule

    #########################################################
    # Forward
    #########################################################
    def forward(self, batch):
        """
        @brief run MKLag on a batch and compute the losses. 
                Videos are processed one by one inside mklag.
        
        return: evals (losses + GT needed by the metrics), batch (updated with predictions)
        """
        lag_labels = {k: batch[v] for k, v in LAG_KEYS.items()}
        res = self.model(
            batch['embedding'],
            batch['ydev'],
            batch['ycat'],
            lag_labels,
            compute_loss=True,
        )

        evals = {
            'loss': res['loss'],  # required for backprop in Lightning
            'scene_loss': res['scene_loss'],
            'lag_loss': res['lag_loss'],
        }


        batch['pred_lag'] = res['lag_probs'].detach()        # [B, 9] = [z_k, s_k, d_k] x 3 = [presence score, pred start, pred duration]
        batch['pred_dev'] = res['frame'].detach()            # [B, T]
        batch['pred_cat'] = res['scene'].detach()          # [B, 1]
        batch['scene_logits'] = res['scene_logits'].detach()
        batch['scene_features'] = res['scene_features'].detach()

        for k in GT_LAG_FIELDS:
            evals[k] = batch[k]
        evals['scene_logits'] = batch['scene_logits']
        evals['scene_features'] = batch['scene_features']
        return evals, batch

    def evaluations(self, batch: dict, evals: dict):
        evals['loss2log'] = evals['loss']

        evals['pred_lag'] = batch['pred_lag']
        evals['pred_dev'] = rearrange(batch['pred_dev'], "b t -> (b t)")
        evals['pred_cat'] = batch['pred_cat'].flatten()

        evals['ydev'] = rearrange(batch['ydev'], "b t -> (b t)")
        evals['ycat'] = batch['ycat'].flatten()

        evals['accs_dev'] = (evals['pred_dev'] == evals['ydev']).float()
        evals['accs_cat'] = (evals['pred_cat'] == evals['ycat']).float()

        # NB: for the prediction pickle the batch size is 1
        evals['video_name'] = batch['video_name'][0]

    def logging(self, evals: dict, step_label: str):
        batch_size = len(evals['ycat'])
        for k in ['loss2log', 'scene_loss', 'lag_loss', 'accs_dev', 'accs_cat']:
            evals[k] = evals[k].detach()
            self.log(f'{step_label}/{k}', evals[k].mean(), on_epoch=True, on_step=True, batch_size=batch_size)

    def get_evals(self, batch: dict, step: str):
        evals, batch = self.forward(batch)
        self.evaluations(batch, evals)

        if step not in ['test', 'pred']:
            self.logging(evals, step)
        return evals

    #########################################################
    # Steps
    #########################################################
    def training_step(self, batch, batch_idx):
        return self.get_evals(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.get_evals(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.get_evals(batch, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        evals = self.get_evals(batch, 'pred')
        # everything on CPU / without graph for the prediction pickle
        return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in evals.items()}

    #########################################################
    # Optimizer & Scheduler
    #########################################################
    def configure_optimizers(self):
        optimizer = build_optimizer(self.parameters(), self.optimizer_name, self.lr, self.l2)
        print(f'Optimizer : {optimizer}')

        max_epochs = getattr(getattr(self.cfg, 'trainer', None), 'max_epochs', 100)
        scheduler = build_scheduler(optimizer, self.schedule, self.lr, max_epochs, monitor="train/loss2log")

        d = {'optimizer': optimizer}
        if scheduler is not None:
            d['lr_scheduler'] = scheduler
        return d
