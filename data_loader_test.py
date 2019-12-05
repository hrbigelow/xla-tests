import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl


class VirtualBatch(object):
    def __init__(self, dataset):
        super(VirtualBatch, self).__init__()
        self.ds = dataset
        self.wav_dec_input = torch.empty((self.ds.batch_size,
            self.ds.window_batch_size))

    def __repr__(self):
        fmt = (
            'wav_dec_input.shape: {}\n'
        )
        return fmt.format(self.wav_dec_input.shape)

    def populate(self):
        """
        sets the data for one sample in the batch
        """
        rg = torch.empty((self.ds.batch_size), dtype=torch.int64).cpu()
        picks = rg.random_() % 1000 

        for b, wi in enumerate(picks):
            self.wav_dec_input[b,...] = torch.empty(self.ds.window_batch_size)

        assert self.wav_dec_input.shape[0] == 8


    def to(self, device):
        self.wav_dec_input = self.wav_dec_input.to(device)
        assert self.wav_dec_input.shape[0] == 8


class Slice(torch.utils.data.IterableDataset):
    """
    Defines the current batch of data in iterator style.
    Use with automatic batching disabled, and collate_fn = lambda x: x
    """
    def __init__(self, batch_size, window_batch_size):
        self.init_args = {
                'batch_size': batch_size,
                'window_batch_size': window_batch_size
                }
        self._initialize()


    def _initialize(self):
        super(Slice, self).__init__()
        self.target_device = None
        self.__dict__.update(self.init_args)


    def __setstate__(self, init_args):
        self.init_args = init_args 
        self._initialize()


    def __getstate__(self):
        return self.init_args


    def init_geometry(self):
        """
        Initializes:
        self.enc_in_len
        self.trim_ups_out
        self.trim_dec_out
        self.trim_dec_in
        """
        # Calculate max length of mfcc encoder input and wav decoder input
        w = self.window_batch_size
        mfcc_vc = self.mfcc_vc
        beg_grcc_vc = self.decoder_vcs['beg_grcc']
        end_grcc_vc = self.decoder_vcs['end_grcc']
        end_ups_vc = self.decoder_vcs['last_upsample']
        end_enc_vc = self.encoder_vcs['end']

        do = vconv.GridRange((0, 100000), (0, w), 1)
        di = vconv.input_range(beg_grcc_vc, end_grcc_vc, do)
        ei = vconv.input_range(mfcc_vc, end_grcc_vc, do)
        mi = vconv.input_range(mfcc_vc.child, end_grcc_vc, do)
        eo = vconv.output_range(mfcc_vc, end_enc_vc, ei)
        uo = vconv.output_range(mfcc_vc, end_ups_vc, ei)

        # Needed for trimming various tensors
        self.enc_in_len = ei.sub_length()
        self.enc_in_mel_len = mi.sub_length()
        self.emb_len = eo.sub_length() 
        self.dec_in_len = di.sub_length()
        self.trim_dec_in = torch.tensor([di.sub[0] - ei.sub[0], di.sub[1] -
            ei.sub[0]], dtype=torch.long)
        self.trim_ups_out = torch.tensor([di.sub[0] - uo.sub[0], di.sub[1] -
            uo.sub[0]], dtype=torch.long)
        self.trim_dec_out = torch.tensor([do.sub[0] - di.sub[0], do.sub[1] -
            di.sub[0]], dtype=torch.long)

        # Generate slices from input
        self.in_start = []
        for sam in self.samples:
            for b in range(sam.wav_b, sam.wav_e - w, w):
                self.in_start.append((b, sam.voice_index))


    def post_init(self, encoder_vcs, decoder_vcs):
        """
        Initializes:
        self.slices
        Depends on information computed from the model, so must be
        called after model construction.
        """
        self.encoder_vcs = encoder_vcs
        self.decoder_vcs = decoder_vcs
        self.init_geometry()



    def set_target_device(self, target_device):
        self.target_device = target_device
        # self.trim_dec_in = self.trim_dec_in.to(target_device)
        # self.trim_ups_out = self.trim_ups_out.to(target_device)
        # self.trim_dec_out = self.trim_dec_out.to(target_device)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get a random slice of a file, together with its start position and ID.
        Random state is from torch.{get,set}_rng_state().  It is on the CPU,
        not GPU.
        """
        vb = VirtualBatch(self)
        vb.populate()

        if self.target_device:
            vb.to(self.target_device)

        return vb 


class WavLoader(torch.utils.data.DataLoader):
    """
    Data loader which may be wrapped by a
    torch_xla.distributed.parallel_loader.
    This loader returns batches of tensors on cpu, optionally
    pushing them to target_device if provided
    """
    @staticmethod
    def ident(x):
        return x

    def __init__(self, wav_dataset, target_device=None):
        self.target_device = target_device
        super(WavLoader, self).__init__(
                dataset=wav_dataset,
                batch_sampler=None,
                collate_fn=self.ident
                )

    def set_target_device(self, target_device):
        self.dataset.set_target_device(target_device)



class TPULoaderIter(object):
    def __init__(self, device):
        dataset = Slice(10, 1000)
        data_loader = WavLoader(dataset)
        para_loader = pl.ParallelLoader(data_loader, [device])
        self.per_dev_loader = para_loader.per_device_loader(device)

    def __next__(self):
        return self.per_dev_loader.__next__()[0]


def main():
    tpu_iter = TPULoaderIter(xm.xla_device())
    vb1 = next(tpu_iter)
    vb2 = next(tpu_iter)
    


if __name__ == '__main__':
    main()

