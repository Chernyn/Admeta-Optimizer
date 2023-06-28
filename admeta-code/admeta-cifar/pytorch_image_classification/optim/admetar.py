import math
import torch
from torch.optim.optimizer import Optimizer


class AdmetaR(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False, rectify=False, lamda=0.2, k=6):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= lamda < 1.0:
            raise ValueError("Invalid lamda parameter at index 1: {}".format(lamda))
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)],lamda=lamda, k=k)
        super(AdmetaR, self).__init__(params, defaults)
        self.rectify = rectify


    def __setstate__(self, state):
        super(AdmetaR, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('AdmetaR does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['I_avg'] = torch.zeros_like(p_data_fp32)
                    state['g_1'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['g_1'] = state['g_1'].type_as(p_data_fp32)
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    state['I_avg'] = state['I_avg'].type_as(p_data_fp32)

                I_avg = state['I_avg']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                lamda = group['lamda']

                state['step'] += 1

                if state['step'] == 1:
                    state['g_1'].copy_(grad)
                lamda_t = lamda ** state['step']

                I_avg.mul_(lamda).add_(grad)
                h_avg = grad.mul(10.0/lamda - 9.0).add(25.0 - 10.0 * lamda-10 * (1/lamda)).add_(lamda_t * 10.0, state['g_1'])
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, h_avg, h_avg)
                exp_avg.mul_(beta1).add_(1 - beta1, h_avg)

                if self.rectify:
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                            N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1

                        buffered[2] = step_size

                    if N_sma >= 5:
                        if group['weight_decay'] != 0:
                            p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                        p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                        p.data.copy_(p_data_fp32)
                    elif step_size > 0:
                        if group['weight_decay'] != 0:
                            p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                        p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                        p.data.copy_(p_data_fp32)

                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    step_size = math.sqrt(1.0 - beta2 ** state['step']) / (1.0 - beta1 ** state['step'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=(-group['lr'] * group['weight_decay']))


                # alpha_t = 0.5*(1.0+1.0/(0.01*(torch.tensor(state['step'])).sqrt()+1.0))
                alpha_t = 0.8 * (1.0 + 1.0 / (0.1 * (torch.tensor(state['step'])).sqrt() + 3.8))
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(alpha_t, p.data - slow_p)
                    p.data.copy_(slow_p)
        return loss