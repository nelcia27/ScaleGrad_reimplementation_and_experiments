import scaleGrad
from transformers import Trainer


gamma = 0.2
class ScaleGradTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = scaleGrad.sg_loss(inputs, gamma, outputs)
        return (loss, outputs) if return_outputs else loss