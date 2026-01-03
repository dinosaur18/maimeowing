from transformers import TrainerCallback

from .mask import GTNMMask


class AnnealingCallback(TrainerCallback):
    
    def __init__(
        self,
        logit_scaling_init: float,
        logit_scaling_max: float,
        tau_init: float,
        tau_min: float,
        num_steps: int,
    ):
        super().__init__()
        
        self.logit_scaling_init = logit_scaling_init
        self.logit_scaling_max = logit_scaling_max

        self.tau_init = tau_init
        self.tau_min = tau_min
        
        self.num_steps = num_steps

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        
        r = state.global_step / self.num_steps
        logit_scaling = self.logit_scaling_init * (1 - r) + self.logit_scaling_max * r
        tau = self.tau_init * (1 - r) + self.tau_min * r

        GTNMMask.logit_scaling = logit_scaling
        GTNMMask.tau = tau
