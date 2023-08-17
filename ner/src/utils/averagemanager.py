import contextlib
import torch

class AverageManager(torch.nn.Module):
    def __init__(self, var_list, use_moving_average=True, moving_average=0.999):
        super(AverageManager, self).__init__()
        self.state = "temporal"
        self.use_moving_average = torch.nn.Parameter(torch.BoolTensor([use_moving_average])[0], requires_grad=False)
        if self.use_moving_average:
            assert 0.0 < moving_average < 1.0
            self.moving_average = torch.nn.Parameter(torch.tensor(moving_average))
        else:
            self.moving_average = torch.nn.Parameter(torch.tensor(0.0))

        self.var_list = list(var_list)
        self.average_var_list = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(var), requires_grad=False) for var in self.var_list])
        self.num_updated = torch.nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)

    def _get_biased_averages(self):
        if self.use_moving_average:
            if self.num_updated == 0:
                return self.average_var_list
            else:
                moving_average_bias = 1.0 - (self.moving_average ** self.num_updated.to(torch.float))
                return [average_var / moving_average_bias for average_var in self.average_var_list]
        else:
            return self.average_var_list

    def step(self):
        if self.use_moving_average:
            average_fraction = self.moving_average
            temporal_fraction = 1.0 - self.moving_average
        else:
            float_num_updated = self.num_updated.to(torch.float)
            average_fraction = float_num_updated / (float_num_updated + 1.0)
            temporal_fraction = 1.0 / (float_num_updated + 1.0)

        for average_var, temporal_var in zip(self.average_var_list, self.var_list):
            average_var.data = average_var * average_fraction + temporal_var * temporal_fraction

        self.num_updated.add_(1)

    @contextlib.contextmanager
    def average_context(self):
        self.use_average()
        try:
            yield
        finally:
            self.use_temporal()

    def use_average(self):
        assert self.state == "temporal"
        self.cache = [temporal_var.clone().detach() for temporal_var in self.var_list]
        for average_var, temporal_var in zip(self._get_biased_averages(), self.var_list):
            temporal_var.data = average_var
        self.state = "average"
        return
    def use_temporal(self):
        assert self.state == "average"
        for temporal_var, cache_value in zip(self.var_list, self.cache):
            temporal_var.data = cache_value
        del self.cache
        self.state = "temporal"
        return


