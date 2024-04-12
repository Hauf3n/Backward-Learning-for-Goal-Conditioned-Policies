import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8

def goal_policy(env, state, policies, key):
    if key == K_1:
        return np.random.randint(env.action_space.n)
    else:
        policy_number = key - K_2
        policy = policies[policy_number]
        action = policy.act(state)
        return action

def control_policies_with_keyboard(env, policies):
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()

    # Define a mapping from key codes to policies
    num_learned_policies = len(policies)
    valid_learned_policy_keys = [(i)+K_2 for i in range(num_learned_policies)]
    keys = [K_1] + valid_learned_policy_keys

    # Define the environment interaction loop
    abort = False
    state = env.reset() 
    last_key_event = K_1 # random policy at start
    while not abort:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                abort = True
            elif event.type == KEYDOWN and event.key in keys:
                last_key_event = event.key

        action = goal_policy(env, state, policies, last_key_event)
        next_state, reward, done, _ = env.step(action)
        env.render()
        pygame.display.update()
        state = next_state
        clock.tick(20)

    pygame.quit()
