import torch
import torch.nn.functional as F


class GradCAMPlusPlus:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks()

    def _hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output  # [B, K, h, w]

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]  # [B, K, h, w]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # [1, C]
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        grads = self.gradients.detach()
        acts = self.activations.detach()

        grad2 = grads ** 2
        grad3 = grads ** 3

        denom = 2.0 * grad2 + (acts * grad3).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))

        alpha = grad2 / (denom + 1e-8)
        relu_grads = F.relu(grads)
        weights = (alpha * relu_grads).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1)[0]
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu()


def densenet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    return model.features.denseblock4
