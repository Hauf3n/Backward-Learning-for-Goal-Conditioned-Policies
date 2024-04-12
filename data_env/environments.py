from __future__ import annotations
import numpy as np
import copy
import random
import pygame
import gymnasium as gym
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_1, K_2, K_3, K_4, K_5
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def collect_data(env, timesteps):
    state_seqs = []
    action_seqs = []
    rewards_seqs = []
    seq_lengths = []

    # reset
    observation = env.reset()

    # init seq collection
    states = [observation.copy()]
    actions = []
    rewards = []

    for i in range(timesteps-1):
        action = env.action_space.sample()
        observation, reward, terminated, info = env.step(action)

        # collect
        states.append(observation.copy())
        actions.append(action)
        rewards.append(reward)

        if terminated:
            # save seq
            state_seqs.append(np.array(states,dtype=np.float16))
            action_seqs.append(actions)
            rewards_seqs.append(rewards)
            seq_lengths.append(len(states))

            # reset
            observation = env.reset()
            states = [observation.copy()]
            actions = []
            rewards = []
            
    return state_seqs, action_seqs, rewards_seqs, seq_lengths

class TabularRL(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, img_obs=False, maze=False, maze_seed=1337,delete_wall_p=0.3):
        self.width = width
        self.height = height
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.MultiDiscrete([self.width, self.height])
        self.img_obs = img_obs
        
        self.grid = np.zeros((self.width, self.height), dtype=int)
        self.goals = [(1, 1), (1, width-2), (height-2, 1), (height-2, width-2)]
        self.colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0)]
        
        if maze:
            self.maze = generate_maze(height, width, maze_seed, delete_wall_p)
        self.walls = get_walls(self.maze) if maze else []
        
        self.agent = self.set_agent_position(random_position=True)#[self.width//2, self.height//2]
        self.goal = None
        self.reward_range = (0, 1)

    def step(self, action):
        done = False
        
        if action == 0 and self.agent[0] > 0 and (self.agent[0]-1, self.agent[1]) not in self.walls:
            self.agent[0] -= 1
        elif action == 1 and self.agent[1] > 0 and (self.agent[0], self.agent[1]-1) not in self.walls:
            self.agent[1] -= 1
        elif action == 2 and self.agent[0] < self.width-1 and (self.agent[0]+1, self.agent[1]) not in self.walls:
            self.agent[0] += 1
        elif action == 3 and self.agent[1] < self.height-1 and (self.agent[0], self.agent[1]+1) not in self.walls:
            self.agent[1] += 1
        else:
            pass  # Standing still

        if self.agent == self.goal:
            reward = 1
            #done = True
        elif self.env_steps > 10:
            done = True
            reward = 0
        else:
            reward = 0
            
        self.env_steps += 1
        info = {}
        
        if not self.img_obs:
            return np.array([self.agent[0]/self.width,self.agent[1]/self.height]), np.array(reward), done, info
        else:
            return self.get_rgb_image(), np.array(reward), done, info
        
    
    def get_state(self):
        return np.array([self.agent[0]/self.width,self.agent[1]/self.height])
            
    def reset(self):
        self.set_agent_position(random_position=True)
        self.goal = self.goals[np.random.randint(0, len(self.goals))]
        self.env_steps = 0
        
        if not self.img_obs:
            return np.array([self.agent[0]/self.width,self.agent[1]/self.height])
        else:
            return self.get_rgb_image()
    
    def set_agent_position(self, x=1, y=1, random_position=False):
        
        if random_position or (x,y) in self.walls:
            new_x = int(random.random()*self.width)
            new_y = int(random.random()*self.height)
            self.set_agent_position(new_x, new_y, random_position=False)   
        else:
            self.agent = [x, y]
            
    def is_wall(self, x, y):
        if (x,y) in self.walls:
            return True
        else:
            return False
        
    def get_rgb_image(self):
        scale = 10
        width = self.width * scale
        height = self.height * scale

        # Create a blank RGB image
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill the image with black color
        rgb_image.fill(0)

        # Convert the numpy array to a pygame Surface
        surface = pygame.surfarray.make_surface(rgb_image)

        # Draw walls and goals
        for x, y in self.walls:
            rect = pygame.Rect(x * scale, y * scale, scale, scale)
            pygame.draw.rect(surface, (255, 255, 255), rect)

        # Draw agent
        #pygame.draw.circle(surface, (0, 255, 0), (self.agent[0] * scale + scale // 2, self.agent[1] * scale + scale // 2), scale // 3)
        agent_size = int(scale * 0.6)  # Adjust the scaling factor as needed
        agent_x = self.agent[0] * scale + (scale - agent_size) // 2
        agent_y = self.agent[1] * scale + (scale - agent_size) // 2
        pygame.draw.rect(surface, (0, 255, 0), (agent_x, agent_y, agent_size, agent_size))

        # Convert the pygame Surface back to a numpy array
        surface = pygame.transform.scale(surface, (64, 64))
        rgb_image = pygame.surfarray.array3d(surface)
        
        return np.float16((np.moveaxis(rgb_image, 0, 1)/255).transpose(2,0,1))
    
    def render(self, mode='human'):
        if mode == 'human':
            scale = 30
            screen = pygame.display.set_mode((self.width*scale, self.height*scale))
            screen.fill((0, 0, 0))

            # Draw walls and goals
            for x, y in self.walls:
                rect = pygame.Rect(x*scale, y*scale, scale, scale)
                pygame.draw.rect(screen, (255, 255, 255), rect)
            for i, (x, y) in enumerate(self.goals):
                color = self.colors[i]
                rect = pygame.draw.rect(screen, color, (x*scale, y*scale, scale, scale))

            # Draw agent
            pygame.draw.circle(screen, (0, 255, 0), (self.agent[0]*scale+scale//2, self.agent[1]*scale+scale//2), scale//3)

            pygame.display.flip()

        elif mode == 'rgb_array':
            raise NotImplementedError
            
    def forward(self, state, action):
        if action == 0 and state[0] > 0 and (state[0]-1, state[1]) not in self.walls:
            new_state = [state[0]-1, state[1]]
        elif action == 1 and state[1] > 0 and (state[0], state[1]-1) not in self.walls:
            new_state = [state[0], state[1]-1]
        elif action == 2 and state[0] < self.width-1 and (state[0]+1, state[1]) not in self.walls:
            new_state = [state[0]+1, state[1]]
        elif action == 3 and state[1] < self.height-1 and (state[0], state[1]+1) not in self.walls:
            new_state = [state[0], state[1]+1]
        else:
            new_state = [state[0], state[1]]  # Standing still

        return new_state
    
    def backward(self, state, action):
        if action == 0 and state[0] < self.width-1 and (state[0]+1, state[1]) not in self.walls:
            new_state = [state[0]+1, state[1]]
        elif action == 1 and state[1] < self.height-1 and (state[0], state[1]+1) not in self.walls:
            new_state = [state[0], state[1]+1]
        elif action == 2 and state[0] > 0 and (state[0]-1, state[1]) not in self.walls:
            new_state = [state[0]-1, state[1]]
        elif action == 3 and state[1] > 0 and (state[0], state[1]-1) not in self.walls:
            new_state = [state[0], state[1]-1]
        else:
            new_state = [state[0], state[1]]  # Standing still

        return new_state
    
    def planning(self, state):
        distance = np.full((self.width, self.height), np.inf)
        distance[state[0], state[1]] = 0

        visited = set()
        visited.add((state[0], state[1]))

        queue = []
        queue.append(state)

        while len(queue) > 0:
            curr_state = queue.pop(0)

            for action in range(5):
                next_state = self.forward(curr_state, action)

                if tuple(next_state) not in visited:
                    visited.add(tuple(next_state))

                    if next_state == curr_state:
                        cost = 0
                    else:
                        cost = 1

                    if distance[next_state[0], next_state[1]] > distance[curr_state[0], curr_state[1]] + cost:
                        distance[next_state[0], next_state[1]] = distance[curr_state[0], curr_state[1]] + cost
                        queue.append(next_state)

        return distance
    
    def evaluate_policy(self, policy, goal, trails=5, allowed_steps_factor=2):
        
        
        sp_grid = self.planning((int(goal[0]*self.width),int(goal[1]*self.height))).transpose()
        print("goal position: ", (int(goal[0]*self.width),int(goal[1]*self.height)))
        eval_grid = np.zeros((self.width, self.height))
        
        for x in range(self.width):
            for y in range(self.height):
                if not self.is_wall(x,y):
                    
                    shortest_path = sp_grid[y,x]
                    allowed_steps = shortest_path * allowed_steps_factor
                    if allowed_steps == np.inf:
                        continue
                    for trail in range(trails):
                        
                        timestep = 0
                        keep_running = True
                        self.reset()
                        self.set_agent_position(x,y)
                        s = self.get_rgb_image()
                        #print(allowed_steps)
                        while timestep < allowed_steps and keep_running:  
                            
                            action = policy.act(s)
                            s, _, _, _ = self.step(action)
                            
                            if tuple(self.get_state()) == goal:
                                keep_running = False
                                eval_grid[y,x] += 1
                                
                            timestep += 1
                else:
                    eval_grid[y,x] = -1
                    
        eval_grid[int(goal[1]*self.width),int(goal[0]*self.height)] = -2
                                
        return eval_grid
        
        
    def evaluate_policy_multigoal(self, policy, goals, trails=5, allowed_steps_factor=2):
        
        sp_grids = []
        for i in range(len(goals)):
            goal = goals[i]
            sp_grid = self.planning((int(goal[0]*self.width),int(goal[1]*self.height))).transpose()
            sp_grids.append(sp_grid)
        
        eval_grid = -3 * np.ones((self.width, self.height))
        reached_closest_goal = 0

        for x in range(self.width):
            for y in range(self.height):
                if not self.is_wall(x,y):

                    shortest_paths = []
                    for i in range(len(goals)): 
                        shortest_paths.append(sp_grids[i][y,x])

                    #print(np.array(shortest_paths))    
                    #print(np.array(shortest_paths).shape)  
                    shortest_path = np.min(np.array(shortest_paths))
                    allowed_steps = shortest_path * allowed_steps_factor
                    if allowed_steps == np.inf:
                        continue
                        
                    for trail in range(1):
                        
                        timestep = 0
                        keep_running = True
                        self.reset()
                        self.set_agent_position(x,y)
                        s = self.get_rgb_image()
                        #print(allowed_steps)
                        while timestep < allowed_steps and keep_running:  
                            
                            action = policy.act(s)
                            s, _, _, _ = self.step(action)
                            
                            for i in range(len(goals)):  
                                goal = goals[i]
                                if tuple(self.get_state()) == goal:
                                    keep_running = False
                                    eval_grid[y,x] = i
                                    if shortest_paths[i] == shortest_path:
                                        reached_closest_goal += 1
                                
                            timestep += 1
                else:
                    eval_grid[y,x] = -1
                    
        eval_grid[int(goal[1]*self.width),int(goal[0]*self.height)] = -2
                                
        return eval_grid, reached_closest_goal
    
        
    def test_policy_distance_to_goal(self, goal, args, add=False):

        visited_states = []
        for traj in range(args.sim_trajs):

                self.reset()
                self.set_agent_position(goal[0],goal[1])

                for step in range(args.sim_steps):

                    state,_,_,_ = self.step(np.random.randint(self.action_space.n))
                    visited_states.append((state[0],state[1])) 

        maze = copy.deepcopy(self.maze)
        for y in range(self.width):
            for x in range(self.height):
                if maze[x][y] == '#':
                    maze[x][y] = -1
                else:
                    maze[x][y] = 0
                    
        maze = np.array(maze)            
        for x,y in visited_states:
            x,y = int(x*self.width), int(y*self.height)
            if add:
                maze[x,y] += 1
            else:
                maze[x,y] = 1
                
        return maze
            
                        
def draw_evaluated_policy(maze):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Get the maze dimensions
    height, width = maze.shape

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect('equal')

    # Define the colormap
    cmap = cm.get_cmap('Greens')

    # Normalize the maze values to [0, 1]
    norm = plt.Normalize(vmin=np.min(maze), vmax=np.max(maze))

    # Loop through each cell in the maze
    for i in range(width):
        for j in range(height):
            cell = maze[j, i]
            
            # Check the value of the cell
            if cell == -1:
                # If the cell is a wall, draw a black square
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white'))
            if cell == -2:
                # If the cell is a goal, draw a blue square
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='blue'))   
            if cell == 0:
                # If the cell is a wall, draw a black square
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='red'))
            else:
                # If the cell is not a wall, compute the color based on the value
                color = cmap(norm(cell))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, alpha=0.8))

    # Set the limits of the plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()
    
def draw_evaluated_policy_rgb(maze):
    # Get the maze dimensions
    height, width = maze.shape

    # Create an empty RGB array to store the colors
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the colormap
    cmap = cm.get_cmap('Greens')

    # Normalize the maze values to [0, 1]
    norm = plt.Normalize(vmin=np.min(maze), vmax=np.max(maze))

    # Loop through each cell in the maze
    for i in range(width):
        for j in range(height):
            cell = maze[j, i]
            
            # Check the value of the cell
            if cell == -1:
                # If the cell is a wall, set black color
                rgb_array[j, i] = [255, 255, 255]  # White
            elif cell == -2:
                # If the cell is a goal, set blue color
                rgb_array[j, i] = [0, 0, 255]  # Blue
            elif cell == 0:
                # If the cell is a wall, set red color
                rgb_array[j, i] = [255, 0, 0]  # Red
            else:
                # If the cell is not a wall, compute the color based on the value
                color = cmap(norm(cell))[:3]  # Extract the RGB values from the colormap
                rgb_array[j, i] = (np.array(color) * 255).astype(np.uint8)  # Scale to [0, 255]

    return rgb_array  
    
def draw_evaluated_policy_multigoal_rgb(maze):
    # Get the maze dimensions
    height, width = maze.shape

    # Create an empty RGB array to store the colors
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the colormap
    cmap = cm.get_cmap('Greens')

    # Normalize the maze values to [0, 1]
    norm = plt.Normalize(vmin=np.min(maze), vmax=np.max(maze))

    # Loop through each cell in the maze
    for i in range(width):
        for j in range(height):
            cell = maze[j, i]
            
            # Check the value of the cell
            if cell == -1:
                # If the cell is a wall, set black color
                rgb_array[j, i] = [255, 255, 255]  # White
            elif cell == -2:
                # If the cell is a goal, set blue color
                rgb_array[j, i] = [0, 0, 255]  # Blue
            elif cell == -3:
                # didnt reach a goal
                rgb_array[j, i] = [255, 0, 0]  # Red

            elif cell == 0:
                # reach goal
                rgb_array[j, i] = [255, 128, 0]  # orange
            elif cell == 1:
                # reach goal
                rgb_array[j, i] = [255, 255, 0]  # yellow
            elif cell == 2:
                # reach goal
                rgb_array[j, i] = [0, 255, 255]  # turkis
            elif cell == 3:
                # reach goal
                rgb_array[j, i] = [255, 0, 255]  # purple
            

    return rgb_array 
    
def generate_maze(height, width, seed, delete_wall_p):
    random.seed(seed)  # Set the seed for reproducibility

    # Create a grid filled with walls
    maze = [['#'] * width for _ in range(height)]

    # Initialize the starting position
    start = (1, 1)
    maze[start[0]][start[1]] = ' '

    # Add the starting position to the frontier list
    frontier = [start]

    # List of neighboring cell offsets
    offsets = [(0, 2), (0, -2), (2, 0), (-2, 0)]

    while frontier:
        current = frontier.pop(random.randint(0, len(frontier) - 1))
        x, y = current

        for dx, dy in offsets:
            nx, ny = x + dx, y + dy

            if 0 < nx < height and 0 < ny < width and maze[nx][ny] == '#':
                maze[nx][ny] = ' '
                maze[x + dx // 2][y + dy // 2] = ' '

                # Add the new cell to the frontier list
                frontier.append((nx, ny))

    # Open the exit paths at each corner of the maze
    maze[0][1] = ' '
    maze[height - 2][width - 2] = ' '
    maze[1][width - 2] = ' '
    maze[height - 2][1] = ' '
    maze[1][0] = ' '
    maze[height - 2][width - 2] = ' '
    
    # delete some walls
    for h in range(height):
        for w in range(width):
            if maze[h][w] == '#':
                if random.random() < delete_wall_p:
                    maze[h][w] = ' '
    return maze

def get_walls(maze):
    height = len(maze)
    width = len(maze[0])
    walls = []
    for x in range(width):
        for y in range(height):
            if maze[y][x] == '#':
                walls.append((y,x))
    
    return walls

def show_tabular_environment(env, only_maze_img=False):
    
    if only_maze_img:
        maze = copy.deepcopy(env.maze)
        for y in range(env.width):
            for x in range(env.height):
                if maze[x][y] == '#':
                    maze[x][y] = -1
                else:
                    maze[x][y] = -3
                    
        draw_evaluated_policy(np.rot90(np.rot90(np.rot90(np.transpose(np.array(maze))))))
    else:
        pygame.init()
        screen = pygame.display.set_mode((600, 600))
        clock = pygame.time.Clock()

        # Define the policies
        def random_policy(state, temperature=1.0):
            return np.random.randint(env.action_space.n)

        # Define a mapping from key codes to policies
        key_to_policy = {
            K_1: random_policy
        }

        # Initialize the policy to be used
        current_policy = random_policy
        # Define the environment interaction loop
        abort = False
        state = env.reset() 
        i = 0 
        while not abort:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    abort = True
                elif event.type == KEYDOWN and event.key in key_to_policy:
                    current_policy = key_to_policy[event.key]

            action = current_policy(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            pygame.display.update()
            state = next_state
            clock.tick(20)
            i += 1
            if i % 40 == 0:
                env.set_agent_position(random_position=True)

        pygame.quit()