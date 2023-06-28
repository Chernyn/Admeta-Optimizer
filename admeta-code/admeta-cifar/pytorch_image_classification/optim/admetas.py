import math
import torch
from torch.optim.optimizer import Optimizer


class AdmetaS(Optimizer):

    def __init__(self, params, lr=0.05, beta=0.2, weight_decay=0.0, lamda=0.9, k=6):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if not 0.0 < lamda < 1.0:
            raise ValueError("Invalid lamda parameter at index 0: {}".format(lamda))
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')


        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, lamda=lamda, k=k)
        super(AdmetaS, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(AdmetaS, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('AdmetaS does not support sparse gradients')
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['I_avg'] = torch.zeros_like(p.data)
                    state['g_1'] = torch.zeros_like(p.data)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p.data)
                    state['g_1'] = state['g_1'].type_as(p.data)
                    state['I_avg'] = state['I_avg'].type_as(p.data)
                I_avg = state['I_avg']

                exp_avg = state['exp_avg']
                beta = group['beta']
                lamda = group['lamda']

                state['step'] += 1

                if state['step'] == 1:
                    state['g_1'].copy_(grad)
                lamda_t = lamda ** state['step']

                I_avg.mul_(lamda).add_(grad)
                h_avg = grad.mul(10.0/lamda - 9.0).add(25.0 - 10.0 * lamda-10 * (1/lamda)).add_(lamda_t * 10.0, state['g_1'])

                exp_avg.mul_(beta).add_(1 - beta, h_avg)
                p.data.add_(-group['lr'], exp_avg)

                # alpha_t = 0.5*(1.0+1.0/(0.01*(torch.tensor(state['step']).sqrt())+1.0))
                alpha_t = 0.8 * (1.0 + 1.0 / (0.1 * (torch.tensor(state['step']).sqrt()) + 3.8))
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(alpha_t, p.data - slow_p)
                    p.data.copy_(slow_p)
        return loss