import numpy as np
import torch
import selfies as sf
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pathlib import Path

class SELFIESDataModule():
    def __init__(
        self, 
        batch_size,
        train_data_path,
        validation_data_path,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train = SELFIESDataset(Path(Path.home() / train_data_path))
        self.val   = SELFIESDataset(Path(Path.home() / validation_data_path))

        self.val.vocab     = self.train.vocab
        self.val.vocab2idx = self.train.vocab2idx

        # Drop data from val that we have no tokens for
        self.val.data = [
            smile for smile in self.val.data
            if False not in [tok in self.train.vocab for tok in smile]
        ]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)

# DEFAULT_SELFIES_VOCAB = ['<start>', '<stop>',] + list(sf.ge1t_semantic_robust_alphabet()) + ["[NH1]","[NH1+1]", "[Cl+1]", "[Si]", "[PH1]", "[Se]"] 

class SELFIESDataset(Dataset):
    def __init__(
        self,
        data_type=None,
        fname=None,
        load_data=False,
    ):
        self.data = []
        if load_data:
            assert fname is not None
            with open(fname, 'r') as f:
                selfie_strings = [x.strip() for x in f.readlines()]
            for string in selfie_strings:
                self.data.append(list(sf.split_selfies(string)))
            self.vocab = set((token for selfie in self.data for token in selfie))
            self.vocab.discard(".")
            self.vocab = ['<start>', '<stop>', *sorted(list(self.vocab))]
        else:
            if data_type=="ZINC":
                DEFAULT_SELFIES_VOCAB = ['<start>', '<stop>', '[PH2]', '[NH2+1]', '[\\N+1]', '[#Branch1]', '[C@]', '[\\C@@H1]', '[\\NH1+1]', '[\\O-1]', '[=Branch1]', '[#C]', '[NH1-1]', '[\\C@H1]', '[/O]', '[=P]', '[P+1]', '[=Ring2]', '[\\I]', '[\\NH2+1]', '[#N+1]', '[=Branch2]', '[/S-1]', '[C@@]', '[S@]', '[NH1+1]', '[/N]', '[/NH2+1]', '[Br]', '[=O]', '[/O-1]', '[O]', '[\\S]', '[=NH2+1]', '[-/Ring2]', '[PH1+1]', '[/S]', '[S-1]', '[=SH1+1]', '[S@@]', '[\\N]', '[=C]', '[=P@@]', '[\\N-1]', '[P@]', '[/C@H1]', '[C@H1]', '[=N]', '[=P@]', '[=Ring1]', '[=S+1]', '[C@@H1]', '[C]', '[#Branch2]', '[/C@]', '[S+1]', '[\\S@]', '[CH1-1]', '[Cl]', '[O-1]', '[\\C]', '[/O+1]', '[#N]', '[P@@H1]', '[-\\Ring1]', '[N-1]', '[/Cl]', '[-/Ring1]', '[P@@]', '[CH2-1]', '[/N-1]', '[\\S-1]', '[=S]', '[PH1]', '[/S@]', '[/NH1+1]', '[/C@@]', '[\\F]', '[=S@@]', '[/Br]', '[N+1]', '[Branch2]', '[/C@@H1]', '[/NH1-1]', '[=OH1+1]', '[\\Cl]', '[I]', '[=S@]', '[NH3+1]', '[/NH1]', '[=N-1]', '[S@@+1]', '[F]', '[/N+1]', '[/F]', '[Ring2]', '[Branch1]', '[\\Br]', '[=N+1]', '[\\NH1]', '[NH1]', '[=O+1]', '[S]', '[N]', '[P]', '[=NH1+1]', '[Ring1]', '[/C]', '[\\O]']
                DEFAULT_SELFIES_VOCAB += ['[#Ring1]', '[#Ring2]', '[=NH1]', '[#NH1+1]', '[OH1+1]', 
                                        '[#C@]', '[=C@H1]', '[#C@H1]', '[=C@@H1]', '[#C@@H1]', '[=C@@]', '[#C@@]', '[=C@]', '[=CH1-1]', 
                                        '[SH1]', '[=SH1]','[SH1+1]', '[#SH1+1]', '[#S@]', '[#S@@]', '[=S@@+1]', '[#S@@+1]', 
                                        '[=PH1]', '[#PH1]', '[=PH1+1]', '[#PH1+1]', '[=PH2]', '[#PH2]', '[#P@@]', '[=P@@H1]', '[#P@@H1]', '[#P@]',
                                        ]
            else:
                DEFAULT_SELFIES_VOCAB = ['<start>', '<stop>', '[#Branch1]', '[#Branch2]', 
                    '[#C-1]', '[#C]', '[#N+1]', '[#N]', '[#O+1]', '[=B]', '[=Branch1]', 
                    '[=Branch2]', '[=C-1]', '[=C]', '[=N+1]', '[=N-1]', '[=NH1+1]', 
                    '[=NH2+1]', '[=N]', '[=O+1]', '[=OH1+1]', '[=O]', '[=PH1]', '[=P]', 
                    '[=Ring1]', '[=Ring2]', '[=S+1]', '[=SH1]', '[=S]', '[=Se+1]', '[=Se]', 
                    '[=Si]', '[B-1]', '[BH0]', '[BH1-1]', '[BH2-1]', '[BH3-1]', '[B]', '[Br+2]', 
                    '[Br-1]', '[Br]', '[Branch1]', '[Branch2]', '[C+1]', '[C-1]', '[CH1+1]', 
                    '[CH1-1]', '[CH1]', '[CH2+1]', '[CH2]', '[C]', '[Cl+1]', '[Cl+2]', '[Cl+3]', 
                    '[Cl-1]', '[Cl]', '[F+1]', '[F-1]', '[F]', '[H]', '[I+1]', '[I+2]', '[I+3]', 
                    '[I]', '[N+1]', '[N-1]', '[NH0]', '[NH1+1]', '[NH1-1]', '[NH1]', '[NH2+1]', 
                    '[NH3+1]', '[N]', '[O+1]', '[O-1]', '[OH0]', '[O]', '[P+1]', '[PH1]', '[PH2+1]', 
                    '[P]', '[Ring1]', '[Ring2]', '[S+1]', '[S-1]', '[SH1]', '[S]', '[Se+1]', '[Se-1]', 
                    '[SeH1]', '[SeH2]', '[Se]', '[Si-1]', '[SiH1-1]', '[SiH1]', '[SiH2]', '[Si]', '[=Cl-1]','[OH1+1]', '[=Br-1]', '[#Br-1]', '[=OH0]', '[=SiH1]', '[=I+2]', '[=CH1]', '[=SeH2]', '[=BH2-1]', '[=SiH2]', '[#PH1]', '[=Br+2]', '[=F+1]', '[=NH1]', '[=Cl+3]', '[=SiH2]', '[#SeH2]', '[=I+3]', '[=Se-1]', '[#Se]', '[#Se+1]', '[#NH0]', '[#SiH2]', '[=NH0]', '[=SeH1]', '[#I+2]', '[#CH1]', '[#Cl+2]', '[#Cl+1]', '[#F+1]', '[=SiH1-1]', '[=Si-1]', '[=PH2+1]', '[#Ring1]', '[=Cl+1]', '[#SiH1-1]', '[=CH2+1]', '[#Se-1]', '[#PH2+1]', '[#Si]', '[=Cl+2]', '[#I+3]', '[#NH1+1]', '[#Br+2]', '[#SeH1]', '[=BH0]', '[=CH1+1]', '[=I+1]', '[#CH1+1]', '[=CH2]', '[#BH0]', '[#CH2+1]', '[#I+1]', '[=CH2]', '[#SiH1]', '[#Cl-1]', '[=CH1-1]', '[=BH1-1]', '[=F-1]', '[#Si-1]', '[#F-1]', '[#BH1-1]', '[#Cl+3]', '[#Ring2]',
                ]


            DEFAULT_SELFIES_VOCAB = DEFAULT_SELFIES_VOCAB + list(sf.get_semantic_robust_alphabet() - set(DEFAULT_SELFIES_VOCAB)) + ["[#SH1]", ]
            
            self.vocab = DEFAULT_SELFIES_VOCAB

        self.vocab2idx = {
            v:i
            for i, v in enumerate(self.vocab)
        }
        self.idx2vocab = dict(zip(range(len(self.vocab)), self.vocab))

    def tokenize_selfies(self, selfies_list):   
        tokenized_selfies = []
        for string in selfies_list: 
            tokenized_selfies.append(list(sf.split_selfies(string)))
        return tokenized_selfies 

    def encode(self, smiles, maxl=None):
        if type(smiles[0]) == list:
            maxl = max([len(x) for x in smiles] + ([0] if maxl==None else [maxl]))

            smiles_pad = [x+['<stop>']*(maxl-len(x)) for x in smiles]
            return torch.tensor(np.vectorize(self.vocab2idx.get)(np.array(smiles_pad)))
        else:
            return torch.tensor([self.vocab2idx[s] for s in [*smiles, '<stop>']])

    def decode(self, tokens):
        if tokens.dim() == 2:
            dec = np.vectorize(self.idx2vocab.get)(tokens.cpu())
            dec[torch.tensor(dec=='<stop>').cummax(dim=-1).values] = ''
            return ["".join(x) for x in dec]
        else:
            dec = [self.vocab[t] for t in tokens]
            # Chop out start token and everything past (and including) first stop token
            stop = dec.index("<stop>") if "<stop>" in dec else None # want first stop token
            selfie = dec[0:stop] # cut off stop tokens
            while "<start>" in selfie: # start at last start token (I've seen one case where it started w/ 2 start tokens)
                start = (1+dec.index("<start>")) 
                selfie = selfie[start:]
            selfie = "".join(selfie)
            return selfie

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    @property
    def vocab_size(self):
        return len(self.vocab)

def collate_fn(data):
    # Length of longest molecule in batch 
    max_size = max([x.shape[-1] for x in data])
    return torch.vstack(
        # Pad with stop token
        [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )