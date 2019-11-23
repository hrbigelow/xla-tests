import sys
import torch
from sys import stderr 


class RandDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, in_size):
        self.batch_size = batch_size
        self.in_size = in_size
        self.target_device = None

    def __iter__(self):
        return self

    def __next__(self):
        input = torch.rand(self.batch_size, self.in_size)
        input.detach_()
        input.requires_grad_(False)
        if self.target_device:
            print('RandDataset: moving input to target device', file=stderr)
            stderr.flush()
            input = input.to(self.target_device)
        input.requires_grad_(True)
        return input

    def set_target_device(self, target_device):
        self.target_device = target_device



class RandDataLoader(torch.utils.data.DataLoader):
    """
    Data loader which may be wrapped by a
    torch_xla.distributed.parallel_loader.
    This loader returns batches of tensors on cpu, optionally
    pushing them to target_device if provided
    """
    @staticmethod
    def ident(x):
        return x

    def __init__(self, dataset, target_device=None):
        self.target_device = target_device
        super(RandDataLoader, self).__init__(
                dataset=dataset,
                batch_sampler=None,
                collate_fn=self.ident
                )

    def set_target_device(self, target_device):
        self.dataset.set_target_device(target_device)


class GPULoaderIter(object):
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __next__(self):
        return self.data_iter.__next__()[0]


class TPULoaderIter(object):
    def __init__(self, parallel_loader, device):
        self.per_dev_loader = parallel_loader.per_device_loader(device)

    def __next__(self):
        return self.per_dev_loader.__next__()[0]


def main():
    mode = sys.argv[1]
    batch_size = 5
    in_size = 10
    out_size = 15

    dataset = RandDataset(batch_size, in_size)

    if mode == 'TPU':
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        device = xm.xla_device()
        data_loader = RandDataLoader(dataset, device)
        para_loader = pl.ParallelLoader(data_loader, [device])
        data_iter = TPULoaderIter(para_loader, device)
    else:
        device = torch.device('cuda')
        data_loader = RandDataLoader(dataset)
        data_loader.set_target_device(device)
        data_iter = GPULoaderIter(iter(data_loader))

    layer = torch.nn.Linear(in_size, out_size, True).to(device)
    target = torch.rand(batch_size, out_size).to(device)

    for step in range(10):
        input = next(data_iter)
        output = layer(input)
        loss = ((output - target) ** 2).sum().sqrt() 
        (input_grad,) = torch.autograd.grad(loss, (input,), retain_graph=True)
        loss.backward()
        print(input_grad.std().item())



if __name__ == '__main__':
    main()
