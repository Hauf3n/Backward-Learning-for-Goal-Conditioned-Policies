import wandb
import argparse
import torch
import time

# sweep
wandb_sweep_config = {
    'method': 'random'
}
wandb_metric = {
    'name': 'return_positions',
    'goal': 'maximize' 
}

wandb_sweep_config['metric'] = wandb_metric

# hyperparameters 
wandb_hyperparameter_config = {}
# single value
wandb_hyperparameter_config.update({})
# fixed values
wandb_hyperparameter_config.update({
    'lr_scheduler': {
        'values': ["none","step"]},
})
# range values
wandb_hyperparameter_config.update({
    'lr_wm': {
            'distribution': 'uniform',
            'min': 3e-4,
            'max': 1e-3},
})

wandb_sweep_config['parameters'] = wandb_hyperparameter_config
#pprint.pprint(wandb_sweep_config)
    
def set_wandb_metrics():
    wandb.define_metric("iteration")
    wandb.define_metric("decider_iteration")
    wandb.define_metric("epoch")
    wandb.define_metric("goals")
    wandb.define_metric("lr", step_metric="iteration")
    wandb.define_metric("recon loss", step_metric="iteration")
    wandb.define_metric("categorical divergence", step_metric="iteration")
    wandb.define_metric("backward loss", step_metric="iteration")
    wandb.define_metric("entropy", step_metric="iteration")
    wandb.define_metric("entropy loss", step_metric="iteration")
    wandb.define_metric("epoch time", step_metric="epoch")
    wandb.define_metric("policy", step_metric="goals")
    wandb.define_metric("decider", step_metric="decider_iteration")

def init_run():
    # hyperparamter 
    args = argparse.ArgumentParser()
    
    # pipeline
    args.add_argument('-use_categorical_entropy', type=bool, default=True)
    args.add_argument('-use_categorical_divergence', type=bool, default=True)
    
    args.add_argument('-train_wm', type=bool, default=True)
    args.add_argument('-save_wm', default=False)
    args.add_argument('-load_wm', default=False)

    args.add_argument('-save_data', default=False)
    args.add_argument('-load_data', default=False)
    
    args.add_argument('-goals', type=bool, default=True)
    args.add_argument('-multi_goal', type=bool, default=False)
    
    # representation
    args.add_argument('-categoricals', type=int, default=16)
    args.add_argument('-classes', type=int, default=16)
    args.add_argument('-entropy_coef', type=float, default=0.000035)
    args.add_argument('-min_entropy', type=float, default=0.01)
    args.add_argument('-entropy_start', type=float, default=0.825)
    args.add_argument('-rm_temperature', type=float, default=0.15)

    # world model
    args.add_argument('-lr_wm', type=float, default=3e-4)
    args.add_argument('-batch_size_wm', type=int, default=192)
    args.add_argument('-epochs_wm', type=int, default=333)
    args.add_argument('-action_emb_size', type=int, default=32)
    args.add_argument('-kl_coef', type=float, default=0.0025)
    args.add_argument('-kl_balance_left', type=float, default=0.75)
    args.add_argument('-wm_temperature', type=float, default=0.3)
    
    # env
    args.add_argument('-env_name', default="maze") 
    args.add_argument('-maze_name', default="maze")
    args.add_argument('-num_steps', type=int, default=25000)
    
    # goal
    args.add_argument('-sim_trajs', type=int, default=250)
    args.add_argument('-sim_steps', type=int, default=125)
    args.add_argument('-explore_iterations', type=int, default=5) # start from subgoals
    args.add_argument('-subgoal_visit_threshold', type=int, default=10)
    # eval
    args.add_argument('-eval_trails', type=int, default=3)
    args.add_argument('-eval_len_factor', type=float, default=1.5)
    
    # goal policy
    args.add_argument('-lr_policy', type=float, default=4e-4)
    args.add_argument('-batch_size_policy', type=int, default=64)
    args.add_argument('-epochs_policy', type=int, default=27)
    args.add_argument('-sde_remove_duplicates', type=bool, default=False)
    
    #optimizer
    args.add_argument('-optimizer', default="adamw")# adam adamw
    
    # scheduler
    args.add_argument('-lr_scheduler', default="step")# step cosine plateau cycle 
    
    # training misc
    args.add_argument('-workers', type=int, default=8)
    args.add_argument('-prefetch_factor', type=int, default=2)
    args.add_argument('-device', default="cuda:0")
    args.add_argument('-use_amp', default=True)
    args.add_argument('-hours', type=int, default=12)
    args.add_argument('-minutes', type=int, default=30)
    args.add_argument('-load_model_path', default="_load_model")
    args.add_argument('-load_data_path', default="_load_data")
    
    # wandb
    args.add_argument('--wandb', action='store_false', default=True)
    args.add_argument('-log_every', type=int, default=20)

    args.add_argument('--hyperparameter_search', action='store_true', default=False)

    args.add_argument('-wandb_project_name', default='test')
    args.add_argument('-wandb_entity', default='mahoe')
    args.add_argument('-hyperparameter_runs', type=int, default=150)
    
    # fin
    args = args.parse_args()
    
    # save folder
    vars(args)["project_path"] = "_"+args.wandb_project_name
    
    # device
    vars(args)["device"] = torch.device(args.device)
    vars(args)["dtype"] = torch.float
    
    # env specific
    vars(args)["input_dim"] = 3
    vars(args)["num_actions"] = 5

    minutes = int(args.hours*60) + args.minutes
    script_start_time = time.time()
    script_end_time = script_start_time + (minutes * 60)
    start_time_formatted = time.strftime('%H:%M:%S %d-%m-%Y', time.localtime(script_start_time))
    end_time_formatted = time.strftime('%H:%M:%S %d-%m-%Y', time.localtime(script_end_time))
    print(f"Start: {start_time_formatted} | End: {end_time_formatted}")

    vars(args)["time_over"] = script_end_time    
    
    return args