import copy
import time
import sys

import numpy as np
import torch


mod_name = vars(sys.modules[__name__])['__name__']
if 'adas.' in mod_name:
    from . import adasls_utils as adasls_utils
else:
    import optim.adasls_utils as adasls_utils


class AdaSLS(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch,
                 init_step_size=1,
                 c=0.1,
                 gamma=2.0,
                 eta_max=None,
                 beta=0.99,
                 momentum=0,

                 # gv_option='per_param',
                 # base_opt='adagrad',
                 # pp_norm_method='pp_armijo',

                 # step_size_method='sls',
                 # sls stuff
                 beta_b=0.9,
                 beta_f=2.0,
                 reset_option=1,
                 line_search_fn="armijo",
                 # sps stuff
                 adapt_flag=None,
                 ):
        # NOTE added here to force AdaSLS version
        base_opt = 'adagrad'
        step_size_method = 'sls'
        gv_option = 'per_param'
        pp_norm_method = 'pp_armijo'
        #########################################
        params = list(params)
        super().__init__(params, {})

        self.pp_norm_method = pp_norm_method

        # sps stuff
        self.adapt_flag = adapt_flag

        # sls stuff
        self.beta_f = beta_f
        self.beta_b = beta_b
        self.reset_option = reset_option
        self.line_search_fn = line_search_fn

        # others
        self.params = params
        self.c = c
        self.eta_max = eta_max
        self.gamma = gamma
        self.momentum = momentum
        self.init_step_size = init_step_size
        self.state['step'] = 0
        self.state['step_size_avg'] = 0.
        self.beta = beta
        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.gv_option = gv_option
        self.base_opt = base_opt
        self.step_size_method = step_size_method

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

        # gv options
        self.state['step'] = 0

        self.gv_option = gv_option
        if self.gv_option in ['scalar']:
            self.state['gv'] = 0.

        elif self.gv_option == 'per_param':
            self.state['gv'] = [torch.zeros(
                p.shape).to(p.device) for p in params]

            if self.base_opt in ['amsgrad', 'adam']:
                self.state['mv'] = [torch.zeros(
                    p.shape).to(p.device) for p in params]

            if self.base_opt == 'amsgrad':
                self.state['gv_max'] = [torch.zeros(
                    p.shape).to(p.device) for p in params]

    def step(self, closure, clip_grad=False):
        # increment step
        self.state['step'] += 1

        # deterministic closure
        seed = time.time()

        def closure_deterministic():
            with adasls_utils.random_seed_torch(int(seed)):
                return closure()

        # get loss and compute gradients
        loss, outputs = closure_deterministic()
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)
        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = adasls_utils.get_grad_list(self.params)
        grad_norm = adasls_utils.compute_grad_norm(grad_current)

        # keep track of step
        if self.state['step'] % int(self.n_batches_per_epoch) == 1:
            self.state['step_size_avg'] = 0.

        # if grad_norm < 1e-6:
        #     return 0.

        #  Gv options
        # =============
        if self.gv_option in ['scalar']:
            # update gv
            self.state['gv'] += (grad_norm.item())**2

        elif self.gv_option == 'per_param':
            # update gv
            for i, g in enumerate(grad_current):
                if self.base_opt == 'adagrad':
                    self.state['gv'][i] += g**2

                elif self.base_opt == 'rmsprop':
                    self.state['gv'][i] = (
                        1-self.beta)*(g**2) + (self.beta) * self.state['gv'][i]

                elif self.base_opt in ['amsgrad', 'adam']:
                    self.state['gv'][i] = (
                        1-self.beta)*(g**2) + (self.beta) * self.state['gv'][i]
                    self.state['mv'][i] = (1-self.momentum)*g + \
                        (self.momentum) * self.state['mv'][i]

                else:
                    raise ValueError('%s does not exist' % self.base_opt)

        pp_norm = self.get_pp_norm(grad_current=grad_current)

        # compute step size
        # =================
        step_size = self.get_step_size(
            closure_deterministic, loss,
            params_current, grad_current, grad_norm,
            pp_norm, for_backtracking=False)

        self.try_sgd_precond_update(
            self.params, step_size, params_current, grad_current)
        # save the new step-size
        self.state['step_size'] = step_size

        # compute gv stats
        gv_max = 0.
        gv_min = np.inf
        gv_sum = 0
        gv_count = 0

        for i, gv in enumerate(self.state['gv']):
            gv_max = max(gv_max, gv.max().item())
            gv_min = min(gv_min, gv.min().item())
            gv_sum += gv.sum().item()
            gv_count += len(gv.view(-1))

        self.state['gv_stats'] = {'gv_max': gv_max,
                                  'gv_min': gv_min, 'gv_mean': gv_sum/gv_count}

        self.state['step_size_avg'] += (step_size /
                                        int(self.n_batches_per_epoch))
        self.state['grad_norm'] = grad_norm.item()

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('nans detected')

        return loss, outputs

    def get_pp_norm(self, grad_current):
        if self.pp_norm_method == 'pp_armijo':
            pp_norm = 0

            for i, (g_i, gv_i) in enumerate(zip(
                    grad_current, self.state['gv'])):
                if self.base_opt in ['diag_hessian',
                                     'diag_ggn_ex', 'diag_ggn_mc']:
                    # computing 1 / diagonal for using in the preconditioner
                    pv_i = 1. / (gv_i + 1e-8)

                elif self.base_opt == 'adam':
                    gv_i_scaled = scale_vector(
                        gv_i, self.beta, self.state['step'])
                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                elif self.base_opt == 'amsgrad':
                    self.state['gv_max'][i] = torch.max(
                        gv_i, self.state['gv_max'][i])
                    gv_i_scaled = scale_vector(
                        self.state['gv_max'][i], self.beta, self.state['step'])

                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                elif self.base_opt in ['adagrad', 'rmsprop']:
                    pv_i = 1./(torch.sqrt(gv_i) + 1e-8)
                else:
                    raise ValueError('%s not found' % self.base_opt)

                pp_norm += ((g_i**2) * pv_i).sum()

        elif self.pp_norm_method in ['pp_lipschitz']:
            pp_norm = 0

            for g_i in grad_current:
                if isinstance(g_i, float) and g_i == 0:
                    continue
                pp_norm += (g_i * (g_i + 1e-8)).sum()

        else:
            raise ValueError('%s does not exist' % self.pp_norm_method)

        return pp_norm

    @torch.no_grad()
    def get_step_size(self, closure_deterministic, loss, params_current,
                      grad_current, grad_norm, pp_norm,
                      for_backtracking=False):
        if self.step_size_method == 'fixed_step_size':
            step_size = self.state['step_size']

        if self.step_size_method == 'sps':
            step_size = loss / (self.c * pp_norm + 1e-8)
            if self.adapt_flag in ['constant']:
                if self.eta_max is None:
                    step_size = float(step_size)
                else:
                    step_size = min(self.eta_max, step_size.item())

            elif self.adapt_flag in ['smooth_iter']:
                assert(self.step_size_method != 'fixed_step_size')
                # step_size = loss / (self.c * (grad_norm)**2)
                coeff = self.gamma**(1./self.n_batches_per_epoch)
                # coeff = self.gamma
                if self.state['step'] == 1:
                    step_size = float(step_size)
                else:
                    step_size = min(coeff * self.state['step_size'],
                                    float(step_size))

        elif self.step_size_method == 'sls':
            # reset step size
            # step_size=self.state['step_size']
            step_size = adasls_utils.reset_step(
                step_size=self.state['step_size'],
                n_batches_per_epoch=self.n_batches_per_epoch,
                gamma=self.gamma,
                reset_option=self.reset_option,
                init_step_size=self.init_step_size,
                eta_max=self.eta_max,
                step=self.state['step'])
            # sls line search
            for e in range(100):
                # make potential step
                if self.pp_norm_method == 'pp_lipschitz':
                    adasls_utils.try_sgd_update(self.params, step_size,
                                                params_current, grad_current)
                else:
                    self.try_sgd_precond_update(
                        self.params, step_size,
                        params_current, grad_current, add_momentum=False)

                if for_backtracking:
                    loss_next, outputs = closure_deterministic(
                        for_backtracking=True)
                else:
                    loss_next, outputs = closure_deterministic()

                self.state['n_forwards'] += 1
                found, step_size = self.check_armijo_precond_conditions(
                    step_size,
                    loss, loss_next,
                    grad_norm, pp_norm)

                if found == 1:
                    break

            if found == 0:
                step_size = 1e-6

        return step_size

    def set_step_size(self, step_size):
        assert self.step_size_method == 'fixed_step_size'
        self.init_step_size = step_size

    def check_armijo_precond_conditions(self,
                                        step_size,
                                        loss, loss_next,
                                        grad_norm, pp_norm):
        found = 0

        # computing the new break condition
        if self.gv_option == 'scalar':
            break_condition = loss_next - \
                (loss - (step_size) * self.c *
                 (grad_norm**2) / self.state['gv'])

        elif self.gv_option == 'per_param':
            break_condition = loss_next - \
                (loss - (step_size) * self.c * pp_norm)

        if (break_condition <= 0):
            found = 1

        else:
            # decrease the step-size by a multiplicative factor
            step_size = step_size * self.beta_b

        return found, step_size

    @torch.no_grad()
    def try_sgd_precond_update(self, params, step_size, params_current,
                               grad_current, add_momentum=True):
        if self.gv_option in ['scalar']:
            zipped = zip(params, params_current, grad_current)

            # TODO where 'gv' defined?
            for p_next, p_current, g_current in zipped:
                p_next.data = p_current - \
                    (step_size / torch.sqrt(gv)) * g_current

        elif self.gv_option == 'per_param':
            if self.base_opt == 'adam':
                zipped = zip(params, params_current, grad_current,
                             self.state['gv'], self.state['mv'])
                for p_next, p_current, g_current, gv_i, mv_i in zipped:
                    gv_i_scaled = scale_vector(
                        gv_i, self.beta, self.state['step'])
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if add_momentum is False:
                        mv_i_scaled = (
                            1 - self.momentum) * scale_vector(
                            g_current,
                            self.momentum, self.state['step'])
                    else:
                        mv_i_scaled = scale_vector(
                            mv_i, self.momentum, self.state['step'])

                    p_next.data[:] = p_current.data
                    p_next.data.add_(- step_size, (pv_list * mv_i_scaled))

            elif self.base_opt == 'amsgrad':
                zipped = zip(params, params_current, grad_current,
                             self.state['gv'], self.state['mv'])

                for i, (p_next, p_current, g_current, gv_i, mv_i) in \
                        enumerate(zipped):
                    self.state['gv_max'][i] = torch.max(
                        gv_i, self.state['gv_max'][i])
                    gv_i_scaled = scale_vector(
                        self.state['gv_max'][i], self.beta, self.state['step'])
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if add_momentum is False:
                        mv_i_scaled = (
                            1 - self.momentum) * scale_vector(
                            g_current, self.momentum, self.state['step'])
                    else:
                        mv_i_scaled = scale_vector(
                            mv_i, self.momentum, self.state['step'])

                    p_next.data[:] = p_current.data
                    p_next.data.add_(- step_size, (pv_list * mv_i_scaled))

            elif (self.base_opt in ['rmsprop', 'adagrad']):
                zipped = zip(params, params_current,
                             grad_current, self.state['gv'])
                for p_next, p_current, g_current, gv_i in zipped:
                    pv_list = 1. / (torch.sqrt(gv_i) + 1e-8)

                    p_next.data[:] = p_current.data
                    p_next.data.add_(- step_size, (pv_list * g_current))

            elif (self.base_opt in ['diag_hessian', 'diag_ggn_ex',
                                    'diag_ggn_mc']):
                zipped = zip(params, params_current,
                             grad_current, self.state['gv'])
                for p_next, p_current, g_current, gv_i in zipped:
                    # adding 1e-8 to avoid overflow.
                    pv_list = 1. / (gv_i + 1e-8)

                    p_next.data[:] = p_current.data
                    p_next.data.add_(- step_size, (pv_list * g_current))

            else:
                raise ValueError('%s does not exist' % self.base_opt)
        else:
            raise ValueError('%s does not exist' % self.gv_option)


def scale_vector(vector, alpha, step, eps=1e-8):
    scale = (1-alpha**(max(1, step)))
    return vector / scale
