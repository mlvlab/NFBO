import numpy as np
import torch 


class LatentSpaceObjective:
    '''Base class for any latent space optimization task
        class supports any optimization task with accompanying VAE
        such that during optimization, latent space points (z) 
        must be passed through the VAE decoder to obtain 
        original input space points (x) which can then 
        be passed into the oracle to obtain objective values (y)''' 

    def __init__(
        self,
        xs_to_scores_dict={},
        num_calls=0,
        task_id='',
        ):

        # dict used to track xs and scores (ys) queried during optimization
        self.xs_to_scores_dict = xs_to_scores_dict 
        
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        
        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id


    # def query_oracle(self, x):
    #     ''' Input: 
    #             a single input space item x
    #         Output:
    #             method queries the oracle and returns 
    #             the corresponding score y,
    #             or np.nan in the case that x is an invalid input
    #     '''
    #     raise NotImplementedError("Must implement query_oracle() specific to desired optimization task")
