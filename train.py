import time
import os
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR

from init import *
from data_env.dream_env import *
from data_env.environments import *
from data_env.datasets import *
from utill.util_arguments import *
from utill.util_goals import *
from utill.util_model_shapes import *
from utill.util_reconstruction import *
from goals.sde import *
from goals.learn_policy import *
from networks.policy import *
from networks.representation import *
from networks.world_model import *

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

def main():
    global arguments
    
    # args
    args = init_modify_wandb(arguments)
    
    # environment
    if args.env_name == "maze":
        # load env and collect data
            env = TabularRL(20, 20, img_obs=True, maze=True, maze_seed=123444901, delete_wall_p=0.215)
            state_seqs, action_seqs, rewards_seqs, _ = collect_data(env, args.num_steps)
            vars(args)["num_actions"] = 5

    # data
    dataset = Tabular_Dataset_Backward(state_seqs, action_seqs, rewards_seqs, args.num_actions)
    dataloader = DataLoader(dataset, num_workers=args.workers, batch_size=args.batch_size_wm, shuffle=True, drop_last=True, pin_memory=True,
                           prefetch_factor=args.prefetch_factor, persistent_workers=True)

    # select models
    if args.env_name == "maze":
        rm = Categorical_Representation(args).to(args.device).to(memory_format=torch.channels_last)
        wm = BWM(args).to(args.device)
        
    if args.load_wm:
        rm.load_state_dict(torch.load(args.load_model_path+"rm.pt"))
        wm.load_state_dict(torch.load(args.load_model_path+"wm.pt"))

    else: # train
    
        # optimizer
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(list(wm.parameters()) + list(rm.parameters()), lr=args.lr_wm)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(list(wm.parameters()) + list(rm.parameters()), lr=args.lr_wm)
            
        # lr schedule
        scheduler = None
        if args.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr=8e-5, factor=0.75, patience=(len(dataset)//args.batch_size_wm) * 10, verbose=False)
        elif args.lr_scheduler == "cycle":
            scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=args.epochs_wm, steps_per_epoch=len(dataset)//args.batch_size_wm)
        elif args.lr_scheduler == "step":
            scheduler = StepLR(optimizer, step_size=args.epochs_wm//3, gamma=0.8)
        elif args.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
            
        # regularization for representation entropy 
        minimal_entropy = args.min_entropy
        entropy_start_epoch = int((args.entropy_start * args.epochs_wm))
        entropy_epochs = args.epochs_wm - entropy_start_epoch
        
        # optimize models
        wandb_epoch = 0
        wandb_iteration = 0
        num_batches = 0

        # amp scaler
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        for epoch in tqdm(range(args.epochs_wm)):
                  
            for i, (s,a,t) in enumerate(dataloader):
                

                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=args.use_amp):

                    s = s.to(args.device, non_blocking=True).to(memory_format=torch.channels_last)
                    a = a.to(args.device, non_blocking=True)
                    t = t.to(args.device, non_blocking=True).to(memory_format=torch.channels_last)

                    # categorical representations
                    reconstruction, _, encoding, entropy = rm(s)
                    _, t_distribution, t_encoding = rm.encode(t)

                    # world model predict
                    backward_predictions = wm(torch.cat((encoding, wm.action_embedding(a)), dim=1))

                    #### representation model - reconstruction loss ####
                    if args.env_name == "maze":          
                        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstruction, s, reduction="mean") 
                        
                    representation_loss = reconstruction_loss  
                        
                    #### representation model - entropy loss ####
                    entropy_mean = torch.mean(entropy)
                    if args.use_categorical_entropy:
                        entropy_loss = F.relu(entropy_mean - minimal_entropy)
                        entropy_loss = entropy_loss * int(epoch >= entropy_start_epoch)
                        entropy_loss = (args.entropy_coef * ((epoch - entropy_start_epoch)/entropy_epochs)) * entropy_loss
                          
                    #### world model - kl loss ####
                    if args.train_wm:
                        
                        if args.kl_balance_left == 0.5:
                            
                            kl = t_distribution * (torch.log(t_distribution + 1e-8) - torch.log(backward_predictions + 1e-8))
                            kl_loss = torch.mean(torch.mean(torch.sum(kl, 2), 1))
                            
                        else:
                        
                            kl_left = t_distribution.detach() * (torch.log(t_distribution.detach() + 1e-8) - torch.log(backward_predictions + 1e-8))
                            kl_left = torch.mean(torch.mean(torch.sum(kl_left, 2), 1))
                            
                            kl_right = t_distribution * (torch.log(t_distribution + 1e-8) - torch.log(backward_predictions.detach() + 1e-8))
                            kl_right = torch.mean(torch.mean(torch.sum(kl_right, 2), 1))
                            
                            kl_loss = args.kl_balance_left * kl_left + (1 - args.kl_balance_left) * kl_right
                            
                        wm_loss = args.kl_coef * kl_loss
                    
                    
                    #### total loss ####
                    loss = representation_loss
                    if args.use_categorical_entropy:
                        loss += entropy_loss
                    if args.train_wm:
                        loss += wm_loss
                        
                optimizer.zero_grad(set_to_none=True) 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # scheduler update
                if args.lr_scheduler == "plateau":
                    scheduler.step(reconstruction_loss)
                elif scheduler is not None and args.lr_scheduler != "step":
                    scheduler.step()
                
                # wandb logging
                if num_batches % args.log_every == 0:
                    wandb_iteration += 1  
                    logs = {}
                    if args.train_wm:
                        if args.kl_balance_left == 0.5:
                            logs["backward loss"] = kl_loss
                        else:    
                            logs["backward loss"] = kl_right
                    if args.use_categorical_entropy:
                        logs["entropy loss"] = entropy_loss
                    if args.use_categorical_divergence:
                        with torch.no_grad():
                            cdv = torch.mean( torch.sum((encoding - t_encoding)**2,dim=1)) / 2
                        logs["categorical divergence"] = cdv

                    logs["lr"] = optimizer.param_groups[0]['lr'] 
                    logs["recon loss"] = reconstruction_loss
                    logs["entropy"] = entropy_mean
                    logs["iteration"] = wandb_iteration
                    
                    wandb.log(logs)

                num_batches += 1

            # scheduler update
            if args.lr_scheduler == "step":
                scheduler.step()        

            wandb_epoch += 1

    if args.save_wm:
        torch.save(rm.state_dict(), args.run_path+"rm.pt")
        torch.save(wm.state_dict(), args.run_path+"wm.pt")
        
    rm.eval()
    wm.eval() 

    # visualization
    if True:

        # show image reconstructions
        img = eval_recon_imgs(dataset, rm, args, num_imgs=32)
        img = wandb.Image(zoom(np.clip(img*255, 0, 255).astype(np.uint8), (2, 2, 1), order=3)) # make bigger images
        wandb.log(({"reconstructions":img}))
        
        # show dream videos
        if args.env_name == "maze": 

            env.set_agent_position(1, 1)
            start_obs = env.get_rgb_image()

            dreams = []
            for i in range(3):
                video = dream(env, rm, wm, start_obs, args, steps=150, fps=20)
                dreams.append(wandb.Video(video, fps=20))
            wandb.log(({"dream":dreams}))
    
    # only train encoder & world model
    if not args.goals:
        return

    # set goals
    if args.env_name == "maze":

        
        if args.multi_goal: # multi goal
            multigoals_raw =[[(1,1),(1,env.height-2),(env.width-2,1),(env.width-2,env.height-2)]] # maze corners
            goals = []
            for i in range(len(multigoals_raw)):
                multi_goal = []
                for goalx, goaly in multigoals_raw[i]:
                    env.reset()
                    env.set_agent_position(goalx, goaly)
                    goal_img = env.get_rgb_image()
                    multi_goal.append(goal_img)
                goals.append(multi_goal)

        else: # single goal
            single_goals_raw = [(1,1),(1,env.height-2),(env.width-2,1),(env.width-2,env.height-2)] # maze corners
            goals = []
            for goalx, goaly in single_goals_raw:
                env.reset()
                env.set_agent_position(goalx, goaly)
                goal_img = env.get_rgb_image()
                goals.append([goal_img])
        
    # train goal policy 
    policies = []
    for i, goal in enumerate(goals):
        
        wandb.define_metric(f"policy{i}_iteration")
        wandb.define_metric(f"policy{i}_loss",step_metric=f"policy{i}_iteration")
        wandb.define_metric(f"policy{i}_entropy",step_metric=f"policy{i}_iteration")
        
        policy = learn_policy(goal, env, wm, rm, args, i)
        policy.to(args.device)
        policy.set_encoder(rm, args.device)
        policy.device = args.device
        policies.append(policy)

    # evaluate goal policy
    if args.env_name == "maze":

        if args.multi_goal: # multi goal
            wandb.define_metric(f"multi_goal_returns")
            wandb.define_metric(f"multi_goal_closest")
            eval_grids = []
            for i, policy_goal in enumerate(multigoals_raw):
                p_goals = []
                for goalx, goaly in policy_goal:
                    p_goals.append((goalx/env.width, goaly/env.height))
                grid_perform, closest_goal = env.evaluate_policy_multigoal(policies[0], p_goals, trails=args.eval_trails, allowed_steps_factor=args.eval_len_factor)
                
                mask = grid_perform >= 0
                wandb.log({f"return_positions":np.sum(mask)})

                rgb_img = zoom( draw_evaluated_policy_multigoal_rgb(grid_perform), (45, 45, 1), order=0)
                eval_grids.append(wandb.Image(rgb_img))

            wandb.log({"multi goal policy":eval_grids, 
                       "multi_goal_closest":closest_goal})   
               
        else: # single goal
            eval_grids = []
            returned_to_goal = 0
            for i, policy_goal in enumerate(single_goals_raw):
                wandb.define_metric(f"policy{i}_returns")

                grid_perform = env.evaluate_policy(policies[i], (policy_goal[0]/env.width,policy_goal[1]/env.height), trails=args.eval_trails, allowed_steps_factor=args.eval_len_factor)

                mask = grid_perform > 0
                returned_to_goal += np.sum(grid_perform[mask]) / args.eval_trails
                wandb.log({f"policy{i}_returns":np.sum(grid_perform[mask]) / args.eval_trails})

                rgb_img = zoom( draw_evaluated_policy_rgb(grid_perform), (45, 45, 1), order=0)
                eval_grids.append(wandb.Image(rgb_img, caption=f"{len(policies[i].encountered_states)} States"))

            wandb.log({"policy":eval_grids})
            wandb.log({"return_positions":returned_to_goal})

if __name__ == '__main__':
    
    global arguments
    arguments = init_run()
    
    if arguments.hyperparameter_search:
        sweep_path = os.path.join(os.getcwd(), "sweep.data") 
        if os.path.isfile(sweep_path):
            sweep_id = read_dict_from_file(sweep_path)["sweep_id"]
        else:
            sweep_id = wandb.sweep(wandb_sweep_config, project=arguments.wandb_project_name)
            write_dict_to_file({"sweep_id":sweep_id}, sweep_path)
        
        for i in range(arguments.hyperparameter_runs):
            if time.time() < arguments.time_over:
                wandb.agent(sweep_id, main, project=arguments.wandb_project_name, count=1)
    else:
        main()