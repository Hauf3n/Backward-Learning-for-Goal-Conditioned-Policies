import torch
import numpy as np
import cv2

def dream(env, rm, wm, obs, args, steps=125, fps=25, video_name="dream.mp4", atari=False):
    obs = torch.tensor(np.array(obs), device=args.device, dtype=args.dtype).unsqueeze(0)
    
    _, _, z = rm.encode(obs, temperature=args.rm_temperature)
    
    frames = []
    for _ in range(steps):
        
        # generate actions
        actions = torch.tensor(np.random.randint(env.action_space.n, size=(z.shape[0],1)), device=args.device)
        actions = wm.action_embedding(actions).squeeze(1)
        
        # get prev z
        wm_input = torch.cat((z, actions), dim=1)
        prev_z, _ = wm.sample(wm_input, args.wm_temperature)
        
        # decode prev z
        prev_obs = rm.decode(prev_z.view(1, args.categoricals*args.classes, 1, 1))
        
        frames.append(prev_obs.detach().cpu().numpy())
        
        z = prev_z
    
    if args.save_wm:
        frame_height, frame_width, _ = frames[0][0].transpose(1,2,0).shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # video codec
        out = cv2.VideoWriter(args.run_path+video_name, fourcc, fps, (frame_width, frame_height))
        for frame in frames:
            out.write(np.uint8(frame[0].transpose(1,2,0)))
            
        out.release()
    
    return np.squeeze(np.array(frames, dtype=np.uint8), axis=1)