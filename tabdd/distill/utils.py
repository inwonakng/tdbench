from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.nn as nn


# xavier uniform with controlled random generator
def xavier_uniform(tensor, gain=1.0, generator=None):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    a = np.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-a, a, generator=generator)


def init_weights(m, gain=1.0, generator=None):
    if isinstance(m, nn.Linear):
        xavier_uniform(m.weight, gain=gain, generator=generator)
        m.bias.data.fill_(0.01)


# A simple MLP class
class MLP(nn.Module):
    def __init__(
        self,
        input_shape: int,
        n_hidden_layers: int,
        hidden_dim: int,
        n_labels: int,
    ):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(n_hidden_layers)
            ],
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_labels),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.classifier(self.embedder(x))


def get_mlp(
    input_shape: int,
    n_hidden_layers: int,
    hidden_dim: int,
    n_labels: int,
    random_state: int,
):
    gen = torch.Generator("cpu")
    gen.manual_seed(random_state)
    initializer = partial(init_weights, gain=1.0, generator=gen)

    mlp = MLP(
        input_shape=input_shape,
        n_hidden_layers=n_hidden_layers,
        hidden_dim=hidden_dim,
        n_labels=n_labels,
    ).apply(initializer)
    return mlp


class ReparamModule(nn.Module):
    def _get_module_from_name(self, mn):
        if mn == "":
            return self
        m = self
        for p in mn.split("."):
            m = getattr(m, p)
        return m

    def __init__(self, module):
        super(ReparamModule, self).__init__()
        self.module = module

        param_infos = []  # (module name/path, param name)
        shared_param_memo = {}
        shared_param_infos = (
            []
        )  # (module name/path, param name, src module name/path, src param_name)
        params = []
        param_numels = []
        param_shapes = []
        for mn, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    if p in shared_param_memo:
                        shared_mn, shared_n = shared_param_memo[p]
                        shared_param_infos.append((mn, n, shared_mn, shared_n))
                    else:
                        shared_param_memo[p] = (mn, n)
                        param_infos.append((mn, n))
                        params.append(p.detach())
                        param_numels.append(p.numel())
                        param_shapes.append(p.size())

        assert (
            len(set(p.dtype for p in params)) <= 1
        ), "expects all parameters in module to have same dtype"

        # store the info for unflatten
        self._param_infos = tuple(param_infos)
        self._shared_param_infos = tuple(shared_param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        # flatten
        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        self.register_parameter("flat_param", flat_param)
        self.param_numel = flat_param.numel()
        del params
        del shared_param_memo

        # deregister the names as parameters
        for mn, n in self._param_infos:
            delattr(self._get_module_from_name(mn), n)
        for mn, n, _, _ in self._shared_param_infos:
            delattr(self._get_module_from_name(mn), n)

        # register the views as plain attributes
        self._unflatten_param(self.flat_param)

        # now buffers
        # they are not reparametrized. just store info as (module, name, buffer)
        buffer_infos = []
        for mn, m in self.named_modules():
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    buffer_infos.append((mn, n, b))

        self._buffer_infos = tuple(buffer_infos)
        self._traced_self = None

    def trace(self, example_input, **trace_kwargs):
        assert self._traced_self is None, "This ReparamModule is already traced"

        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)
        example_input = tuple(example_input)
        example_param = (self.flat_param.detach().clone(),)
        example_buffers = (tuple(b.detach().clone() for _, _, b in self._buffer_infos),)

        self._traced_self = torch.jit.trace_module(
            self,
            inputs=dict(
                _forward_with_param=example_param + example_input,
                _forward_with_param_and_buffers=example_param
                + example_buffers
                + example_input,
            ),
            **trace_kwargs,
        )

        # replace forwards with traced versions
        self._forward_with_param = self._traced_self._forward_with_param
        self._forward_with_param_and_buffers = (
            self._traced_self._forward_with_param_and_buffers
        )
        return self

    def clear_views(self):
        for mn, n in self._param_infos:
            setattr(
                self._get_module_from_name(mn), n, None
            )  # This will set as plain attr

    def _apply(self, *args, **kwargs):
        if self._traced_self is not None:
            self._traced_self._apply(*args, **kwargs)
            return self
        return super(ReparamModule, self)._apply(*args, **kwargs)

    def _unflatten_param(self, flat_param):
        ps = (
            t.view(s)
            for (t, s) in zip(flat_param.split(self._param_numels), self._param_shapes)
        )
        for (mn, n), p in zip(self._param_infos, ps):
            setattr(self._get_module_from_name(mn), n, p)  # This will set as plain attr
        for mn, n, shared_mn, shared_n in self._shared_param_infos:
            setattr(
                self._get_module_from_name(mn),
                n,
                getattr(self._get_module_from_name(shared_mn), shared_n),
            )

    @contextmanager
    def unflattened_param(self, flat_param):
        saved_views = [
            getattr(self._get_module_from_name(mn), n) for mn, n in self._param_infos
        ]
        self._unflatten_param(flat_param)
        yield
        # Why not just `self._unflatten_param(self.flat_param)`?
        # 1. because of https://github.com/pytorch/pytorch/issues/17583
        # 2. slightly faster since it does not require reconstruct the split+view
        #    graph
        for (mn, n), p in zip(self._param_infos, saved_views):
            setattr(self._get_module_from_name(mn), n, p)
        for mn, n, shared_mn, shared_n in self._shared_param_infos:
            setattr(
                self._get_module_from_name(mn),
                n,
                getattr(self._get_module_from_name(shared_mn), shared_n),
            )

    @contextmanager
    def replaced_buffers(self, buffers):
        for (mn, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(self._get_module_from_name(mn), n, new_b)
        yield
        for mn, n, old_b in self._buffer_infos:
            setattr(self._get_module_from_name(mn), n, old_b)

    def _forward_with_param_and_buffers(self, flat_param, buffers, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            with self.replaced_buffers(buffers):
                return self.module(*inputs, **kwinputs)

    def _forward_with_param(self, flat_param, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            return self.module(*inputs, **kwinputs)

    def forward(self, *inputs, flat_param=None, buffers=None, **kwinputs):
        flat_param = torch.squeeze(flat_param)
        # print("PARAMS ON DEVICE: ", flat_param.get_device())
        # print("DATA ON DEVICE: ", inputs[0].get_device())
        # flat_param.to("cuda:{}".format(inputs[0].get_device()))
        # self.module.to("cuda:{}".format(inputs[0].get_device()))
        if flat_param is None:
            flat_param = self.flat_param
        if buffers is None:
            return self._forward_with_param(flat_param, *inputs, **kwinputs)
        else:
            return self._forward_with_param_and_buffers(
                flat_param, tuple(buffers), *inputs, **kwinputs
            )
