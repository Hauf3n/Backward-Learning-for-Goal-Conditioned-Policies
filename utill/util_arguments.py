import json
import wandb
import os
from init import *

def init_modify_wandb(arguments):
    args = None
    if arguments.wandb: 
        if arguments.hyperparameter_search:
            wandb.init()
            set_wandb_metrics()
            config = wandb.config
            for key in vars(arguments).keys():
                if key not in config.keys():
                    config[key] = getattr(arguments, key)
            vars(config)["device"] = torch.device(config["device"])
            vars(config)["dtype"] = torch.float
            args = config               
        else:
            wandb.init(project=arguments.wandb_project_name, entity=arguments.wandb_entity, reinit=True)
            set_wandb_metrics()
            args = arguments  
    else:
        wandb.init(project=arguments.wandb_project_name, entity=arguments.wandb_entity, mode="disabled", reinit=True)
        set_wandb_metrics()
        args = arguments
        
    # create folder to save stuff
    if args.save_wm or args.save_data:
        project_path = os.getcwd()+ f'/{args.project_path}'
        run_path = os.getcwd() + f'/{args.project_path}/{wandb.run.name}/'
        if not os.path.isdir(project_path) and args.wandb:
            os.mkdir(project_path)
        if not os.path.isdir(run_path) and args.wandb: 
            os.mkdir(run_path)
        vars(args)["run_path"] = run_path
        
    # create load model folder
    load_model_path = os.getcwd()+ f'/{args.load_model_path}/'
    if not os.path.isdir(load_model_path):
            os.mkdir(load_model_path)
    vars(args)["load_model_path"] = load_model_path

    # create load data folder
    load_data_path = os.getcwd()+ f'/{args.load_data_path}/'
    if not os.path.isdir(load_data_path):
            os.mkdir(load_data_path)
    vars(args)["load_data_path"] = load_data_path
        
    return args
    
def read_dict_from_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def write_dict_to_file(data_dict, filename):
    with open(filename, 'w') as file:
        json.dump(data_dict, file)
