
# Latent Bayesian Optimization via Autoregressive Normalizing Flows
Official PyTorch implementation of the "[Latent Bayesian Optimization via Autoregressive Normalizing Flows](https://arxiv.org/pdf/2504.14889)". (ICLR 2025)

## Installation
- Python 3.7
```bash
pip install -r requirements.yaml
```

We have included a zipped file of the pretrained SeqFlow's checkpoint on 1.27M molecules from the Guacamol dataset.
To unzip the file and prepare it for use, please follow these steps in your terminal:
```bash
zip -s 0 ./model_weight/checkpoint.zip --out ./model_weight/checkpoint_all.zip
unzip ./model_weight/checkpoint_all.zip -d ./model_weight/
```



## How to Run
Below are the commands for running the optimizer with different oracle budgets:

- With an Oracle Budget of 10,000
```Bash
python -m BO.optimizer objective=guacamol objective.task_id=$task_id gpu=$gpu_num wandb=false generative_model=seqflow max_n_oracle_calls=10000 data.oracle_data_load_num=10000 acq_bsz=100 acq_topk=10 use_pretrain=true pretrain_name=1.27M_3 temperature=0.01 dacs_temp=400 train_oracle=200
```

- With an Oracle Budget of 70,000
```Bash
python -m BO.optimizer objective=guacamol objective.task_id=$task_id gpu=$gpu_num wandb=false generative_model=seqflow max_n_oracle_calls=70000 data.oracle_data_load_num=10000 acq_bsz=100 acq_topk=10 use_pretrain=true pretrain_name=1.27M_3 temperature=0.01 dacs_temp=400 train_oracle=200
```

- With an Oracle Budget of 500
```Bash
python -m BO.optimizer objective=guacamol objective.task_id=$task_id gpu=$gpu_num wandb=false generative_model=seqflow max_n_oracle_calls=500 data.oracle_data_load_num=100 acq_bsz=10 acq_topk=5 use_pretrain=true pretrain_name=1.27M_3 temperature=0.01 dacs_temp=400 train_oracle=10
```

## Configuration Parameters
- `gpu`: GPU number to use.
- `objective.task_id`: Task identifier, corresponding to the tasks listed below.
- `train_oracle`: Number of oracle calls after which the generative model is trained.
- `acq_topk`: Number of query points $N_q$ for each trust region
- `acq_bsz`: Batch size of anchor points for each iteration.
- `generative_model.noise`: Standard deviation of the noise in the variational distribution q.
- `generative_model.sim_coef`: Coefficient for the similarity loss.


## Guacamol Tasks
The following table lists the task id and their corresponding task names:

| task_id | Full Task Name     |
|---------|--------------------|
|  adip   | Amlodipine MPO     |
|  med2   | Median molecules 2 |
|  osmb   | Osimertinib MPO    |
|  pdop   | Perindopril MPO    |
|  rano   | Ranolazine MPO     |
|  zale   | Zaleplon MPO       |
|  valt   | Valsartan Smarts   |

## Contact
If you have any questions, please create an issue on this repository or contact at llsshh319@korea.ac.kr.

## Citation
```
@inproceedings{lee2025latent,
  title={Latent Bayesian Optimization via Autoregressive Normalizing Flows},
  author={Lee, Seunghun and Park, Jinyoung and Chu, Jaewon and Yoon, Minseo and Kim, Hyunwoo J},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
