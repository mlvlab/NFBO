import numpy as np
import torch 
import selfies as sf 
from .utils.mol_utils.mol_utils import smiles_to_desired_scores
from .utils.mol_utils.data import SELFIESDataset, collate_fn
from .latent_space_objective import LatentSpaceObjective
from .utils.mol_utils.mol_utils import GUACAMOL_TASK_NAMES
import pkg_resources
# make sure molecule software versions are correct: 
assert pkg_resources.get_distribution("selfies").version == '2.0.0'
assert pkg_resources.get_distribution("rdkit-pypi").version == '2022.3.1'
assert pkg_resources.get_distribution("molsets").version == '0.3.1'

class GuacamolObjective(torch.nn.Module):
    '''GuacamolObjective class supports all guacamol optimization tasks
        and uses the SELFIES VAE by default '''

    def __init__(
        self,
        **kwargs
    ):
        self.mol_objective = MoleculeObjective(
            **kwargs
        )
    
    def selfies_to_tokens(self, selfies):    
        return self.mol_objective.dataobj.tokenize_selfies(selfies)
    
    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        # method assumes x is a single smiles string
        score = smiles_to_desired_scores([x], self.task_id).item()
        

class MoleculeObjective(LatentSpaceObjective):
    '''MoleculeObjective class supports all molecule optimization
        tasks and uses the SELFIES VAE by default '''

    def __init__(
        self,
        task_id='pdop',
        xs_to_scores_dict={},
        max_string_length=128,
        num_calls=0,
        smiles_to_selfies={},
        **kwargs
    ):
        assert task_id in GUACAMOL_TASK_NAMES + ["logp"]

        self.dim                    = 256 # SELFIES VAE DEFAULT LATENT SPACE DIM
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.smiles_to_selfies      = smiles_to_selfies # dict to hold computed mappings form smiles to selfies strings
        self.dataobj = SELFIESDataset(kwargs['data_type'])

        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
        )
        

    def run_oracle(self, x):
        decoded_selfies = [self.dataobj.decode(x[i]) for i in range(x.size(-2))]
        # decode selfies strings to smiles strings (SMILES is needed format for oracle)
        decoded_smiles = []
        for selfie in decoded_selfies:
            smile = sf.decoder(selfie)
            decoded_smiles.append(smile)
            # save smile to selfie mapping to map back later if needed
            self.smiles_to_selfies[smile] = selfie
        scores = smiles_to_desired_scores(decoded_smiles, self.task_id)

        return torch.tensor(scores)

    def query_oracle(self, x_smiles):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        # method assumes x is a single smiles string
        score = smiles_to_desired_scores([x_smiles], self.task_id).item()

        return score
