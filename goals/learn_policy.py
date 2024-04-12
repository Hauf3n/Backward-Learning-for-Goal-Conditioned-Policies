import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from goals.sde import *
from data_env.datasets import *
from networks.policy import *


def learn_policy(goal, env, wm, rm, args, wandb_i):

    print("**** [POLICY] LEARN ****")
    print("[POLICY] Number of goals: ", len(goal))

    state_dict = {} # memorize mapping: argmax encodings to real encodings
    
    sde = SDE() # shortest distance estimator (graph)
    
    # copy goal state(s) to match number of simulations 
    num_goals = len(goal)
    num_repetitions = args.sim_trajs // num_goals
    total_repetitions = num_repetitions * num_goals
    tmp = []
    for i in range(num_goals):
        tmppp = torch.tensor(np.array([goal[i]]))
        tmppp = tmppp.repeat(num_repetitions,1,1,1)
        tmp.append(tmppp)
    goals = torch.cat(tmp, 0)
    
    # encode goals
    _, _, goal_encodings = rm.encode(goals.to(args.device).to(args.dtype), flatten_sample=False, temperature=args.rm_temperature)
    goal_argmax_encodings = torch.argmax(goal_encodings, dim=2).detach().cpu().numpy().astype(str)
    
    goals = set() # used for policy inference
    for i in range(goal_encodings.shape[0]):
        goals.add(str(goal_encodings[i].detach().cpu().numpy()))
    
    # simulate backward trajectories
    trajs = []

    for iteration in range(args.explore_iterations):
        
        if iteration == 0: 
        # start from goals
        
            state_trajs = [goal_encodings.detach().cpu().numpy()]
            state_trajs_estimator = [np.apply_along_axis(join_txt, axis=1, arr=goal_argmax_encodings)]
            action_trajs = []
            state = torch.reshape(goal_encodings, (goal_encodings.shape[0], -1))
            
        else:
        # start from sub-goals (Go-Explore weighted sampling)
            
            # sample subgoals 
            argmax_encodings = sde.get_subgoals(num_subgoals=args.sim_trajs, visit_threshold=args.subgoal_visit_threshold)
            
            encodings = []
            for argmax_encoding in argmax_encodings:
                encodings.append(state_dict[argmax_encoding])
            encodings = torch.tensor(np.array(encodings)).to(args.device)
            argmax_endcodings = torch.argmax(torch.reshape(encodings,(args.sim_trajs,args.categoricals,args.classes)), dim=2).detach().cpu().numpy().astype(str)
            
            state_trajs = [encodings.detach().cpu().numpy()]
            state_trajs_estimator = [np.apply_along_axis(join_txt, axis=1, arr=argmax_endcodings)]
            action_trajs = []
            state = encodings
            
        # memorize mapping: argmax encodings to real encodings
        for i in range(total_repetitions):
            enc = state_trajs[0][i]
            argmax_enc = state_trajs_estimator[0][i]
            state_dict[argmax_enc] = enc
        
        np.random.seed(42)
        for step in range(args.sim_steps):

            # generate actions
            actions = torch.tensor(np.random.randint(env.action_space.n, size=(state.shape[0],1))).to(args.device)
            action_embs = wm.action_embedding(actions).squeeze(1)
            action_trajs.append(actions.detach().cpu().numpy())

            # backward simulation
            wm_input = torch.cat((state.to(args.device), action_embs), dim=1)
            prev_state, prev_state2 = wm.sample(wm_input, args.wm_temperature)

            prev_state_argmax_encoding = torch.argmax(prev_state2, dim=2) 

            # save
            state_trajs.append(prev_state.detach().cpu().numpy())
            state_trajs_estimator.append(np.apply_along_axis(join_txt, axis=1, arr=prev_state_argmax_encoding.detach().cpu().numpy().astype(str)))

            # memorize mapping: argmax encodings to real encodings
            for i in range(total_repetitions):
                enc = state_trajs[step+1][i]
                argmax_enc = state_trajs_estimator[step+1][i]
                state_dict[argmax_enc] = enc

            # next timestep
            state = prev_state
            
        # build up the SDE (graph)
        if iteration == 0:
            sde.add_sequences(np.array(state_trajs_estimator).T.tolist(), goal_first=True)
        else: 
            sde.add_sequences(np.array(state_trajs_estimator).T.tolist(), goal_first=False)
            
        trajs.append((state_trajs,state_trajs_estimator,action_trajs))         
        
    # approximate shortest path values, assume state has categorical nature
    tic = time.time()
    sde.approximate_shortest_distance()
    distance_estimator = sde.get_shortest_distances()
    toc = time.time()
    print("[POLICY] SDE time: ", toc-tic)
    
    # imitation learning
    remove_duplicates = args.sde_remove_duplicates
    imitation_data = {} if remove_duplicates else []
    seen_representation = {} if remove_duplicates else None
    
    # add imitation learning data samples (state,action) to dataset
    for state_trajs,state_trajs_estimator,action_trajs in trajs:
        
        for i in range(len(state_trajs)-1):
            states = state_trajs[i]
            prev_states = state_trajs[i+1]
            actions = action_trajs[i]

            for j in range(states.shape[0]):
                current_state = states[j]
                prev_state = prev_states[j]
                action = actions[j]

                current_state_representation = np.argmax(np.reshape(current_state,(args.categoricals,args.classes)),axis=1)
                prev_state_representation = np.argmax(np.reshape(prev_state,(args.categoricals,args.classes)),axis=1)

                prev_state_representation = str(np.apply_along_axis(join_txt, axis=0, arr=prev_state_representation.astype(str)))
                current_state_representation = str(np.apply_along_axis(join_txt, axis=0, arr=current_state_representation.astype(str)))

                if distance_estimator[prev_state_representation] > distance_estimator[current_state_representation]:
                    if remove_duplicates:

                        key = prev_state_representation
                        if key in seen_representation:
                            visits = seen_representation[prev_state_representation]
                            seen_representation[prev_state_representation] = visits + 1
                        else:
                            seen_representation[prev_state_representation] = 1

                        key += ";"+str(action[0])
                        if key in imitation_data:
                            v = imitation_data[key][2]
                            imitation_data[key] = [prev_state,action[0], v + 1.0]   
                        else:
                            imitation_data[key] = [prev_state,action[0],1.0]

                    else:
                        imitation_data.append([prev_state,action[0],1.0])
    
    if remove_duplicates:
        for seen in seen_representation:
            visits = seen_representation[seen]
            for action in range(args.num_actions):
                key = seen + ";"+str(action)
                if key in imitation_data:
                    value = imitation_data[key]
                    value[2] = value[2] / visits
                    imitation_data[key] = value
            
    if remove_duplicates:
        imitation_data = list(imitation_data.values())
    
    # train a policy on imitation data  
    print("[POLICY] Dataset size:",len(imitation_data))
    
    # random policy
    policy = Goal_Policy(args.categoricals*args.classes, args.num_actions, args.device, args.dtype, args, goals).to(args.device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    
    if len(imitation_data) == 0:
        return policy
        
    # dataset
    imitation_dataset = ImitationDataset(imitation_data)
    imitation_dataloader = DataLoader(imitation_dataset, num_workers=args.workers, batch_size=args.batch_size_policy, shuffle=True, drop_last=True, pin_memory=True, prefetch_factor=args.prefetch_factor, persistent_workers=True)
    
    # amp
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    wandb_iteration = 0
    for _ in tqdm(range(args.epochs_policy)):
        for i, (states, actions, _) in enumerate(imitation_dataloader):
            
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=args.use_amp):
            
                states = states.to(args.device, non_blocking=True)
                actions = actions.to(args.device, non_blocking=True).long()

                action_probs = policy(states)
                imitation_actions = action_probs[torch.arange(args.batch_size_policy),actions]

                imitation_loss = torch.mean(-torch.log(imitation_actions))
            
            policy_optimizer.zero_grad(set_to_none=True) 
            imitation_loss.backward()
            policy_optimizer.step()
            
            wandb_iteration += 1
            wandb.log({f"policy{wandb_i}_loss":imitation_loss,
                       f"policy{wandb_i}_iteration":wandb_iteration})
                     
    return policy