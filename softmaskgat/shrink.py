"""Learned exponent for the fixed ten-fold channel-handling comparison."""

import copy

import numpy as np
import torch

from .routing import routed_forward

SCHEMA = "JNE-SHRINK-v1-canonical-power"
SCHEDULE = {"learning_rate": 0.0005, "weight_decay": 0.0005, "betas": (0.9, 0.999), "gradient_clip": 1.0}


class CanonicalPower(torch.autograd.Function):
    """Compute the quality power with exact endpoints and analytical gradients."""

    @staticmethod
    def forward(ctx, q, exponent):
        if exponent.ndim != 0:
            raise ValueError('One scalar exponent per branch model required')
        value = float(exponent.detach())
        if value == 1.0:
            output = q.clone()
        elif value == 0.0:
            output = torch.ones_like(q)
        else:
            output = q.pow(exponent)
        ctx.save_for_backward(q, exponent, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, exponent, output = ctx.saved_tensors
        grad_q = grad_exponent = None
        if ctx.needs_input_grad[0]:
            grad_q = grad_output * exponent * q.pow(exponent - 1.)
        if ctx.needs_input_grad[1]:
            grad_exponent = (grad_output * (output * q.log())).sum().reshape_as(exponent)
        return grad_q, grad_exponent


class ShrinkAdapter(torch.nn.Module):
    def __init__(self, base_model, binary=False, fixed_lambda=None):
        super().__init__()
        self.base_model = base_model
        self.binary = bool(binary)
        self.fixed_lambda = fixed_lambda
        initial = 1.0 if fixed_lambda is None else float(fixed_lambda)
        if not 0.0 <= initial <= 1.0:
            raise ValueError('lambda must be in [0,1]')

        self.shrink_lambda = torch.nn.Parameter(torch.tensor(initial,
            dtype=torch.float32, device=next(base_model.parameters()).device),
            requires_grad=fixed_lambda is None)

    def config(self):
        return {'schema': SCHEMA, 'binary': self.binary, 'fixed_lambda': self.fixed_lambda,
                'formula': 'where(m,where(m,q,1)**lambda,0)', 'sites_shared': 3,
                'power_numerics': 'canonical exact0/1 values with analytical first derivatives'}

    def base_parameters(self):
        return self.base_model.parameters()

    def effective_quality(self, quality, mask):
        if mask.dtype != torch.bool or quality.shape != mask.shape:
            raise ValueError('q/m shape or mask dtype mismatch')
        if self.shrink_lambda.dtype != torch.float32 or quality.dtype != torch.float32:
            raise TypeError('q and lambda must remain float32, including under autocast')
        if not bool(torch.isfinite(self.shrink_lambda)) or not 0 <= float(self.shrink_lambda.detach()) <= 1:
            raise ValueError('lambda is not finite/in bounds; project only after real updates')
        q = mask.float() if self.binary else quality
        valid = q[mask]
        if not bool(torch.isfinite(valid).all()) or not bool(((valid >= .1) & (valid <= 1)).all()):
            raise ValueError('Valid q must be finite in [0.1,1]')
        safe = torch.where(mask, q, torch.ones_like(q))

        with torch.autocast(device_type=q.device.type, enabled=False):
            powered = CanonicalPower.apply(safe, self.shrink_lambda)
            return torch.where(mask, powered, torch.zeros_like(powered))

    def forward(self, signal, adj, mask, x_anc, quality):
        g = self.effective_quality(quality, mask)

        return routed_forward(self.base_model, signal, adj, mask, x_anc, g, g, g)


class ProjectedAdamW(torch.optim.AdamW):
    """Project the scalar exponent after a successful optimizer step."""
    def __init__(self, model: ShrinkAdapter):
        self.scalar = model.shrink_lambda
        self.actual_updates = 0
        super().__init__([
            {'params': list(model.base_parameters()), 'weight_decay': SCHEDULE['weight_decay']},
            {'params': [self.scalar], 'weight_decay': 0.0},
        ], lr=SCHEDULE['learning_rate'], betas=tuple(SCHEDULE['betas']))

    def step(self, closure=None):
        result = super().step(closure=closure)
        with torch.no_grad():
            self.scalar.clamp_(0., 1.)
        self.actual_updates += 1
        return result

    def state_dict(self):
        value = super().state_dict()
        value['shrink_projection'] = {'schema': SCHEMA, 'actual_updates': self.actual_updates}
        return value

    def load_state_dict(self, state_dict):
        value = copy.deepcopy(state_dict)
        extra = value.pop('shrink_projection')
        if extra['schema'] != SCHEMA:
            raise ValueError('Projection state schema mismatch')
        super().load_state_dict(value)
        self.actual_updates = int(extra['actual_updates'])
        if self.param_groups[1]['weight_decay'] != 0 or self.param_groups[1]['params'][0] is not self.scalar:
            raise ValueError('Scalar optimizer group was not restored faithfully')


def optimization_step(loss, model, optimizer, amp):

    old_scale = float(amp.get_scale())
    old_updates = getattr(optimizer, 'actual_updates', None)
    amp.scale(loss).backward()
    amp.unscale_(optimizer)
    params = list(model.base_parameters()) if isinstance(model, ShrinkAdapter) else list(model.parameters())
    norm = torch.nn.utils.clip_grad_norm_(params, SCHEDULE['gradient_clip'])
    scalar_grad = None
    if isinstance(model, ShrinkAdapter) and model.shrink_lambda.grad is not None:
        scalar_grad = float(model.shrink_lambda.grad)
        model.shrink_lambda.grad.clamp_(-1., 1.)
    amp.step(optimizer)
    amp.update()
    skipped = float(amp.get_scale()) < old_scale
    if old_updates is not None and (optimizer.actual_updates - old_updates) != int(not skipped):
        raise RuntimeError('AMP scale and actual optimizer-step count disagree')
    finite = bool(torch.isfinite(norm)) and (scalar_grad is None or bool(np.isfinite(scalar_grad)))
    if not finite and not skipped:
        raise FloatingPointError('Nonfinite gradient was not skipped by AMP')
    return {'base_gradient_norm': float(norm), 'lambda_gradient_before_clip': scalar_grad,
            'skipped': skipped, 'actual_update': not skipped, 'scale': float(amp.get_scale())}
