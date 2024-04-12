import numpy as np
import heapq

class SDE:
    
    def __init__(self):
        self.goals = set()
        self.distinct_elements = set()
        self.graph = {}
        self.state_dict = {}
        self.found_shortcuts = 0
        self.pq = []
        self.visits = {}

    def add_sequences(self, sequences, goal_first=True):
        
        # update sequences and goals
        if goal_first:
            for sequence in sequences:
                self.goals.add(sequence[0])
                
        for sequence in sequences:
            self.distinct_elements.update(sequence)
        
        # update graph representation
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current_state = sequence[i]
                next_state = sequence[i + 1]
                
                if current_state not in self.graph:
                    self.graph[current_state] = []
                
                if next_state not in self.graph[current_state]:
                    self.graph[current_state].append(next_state)
                    
                # count visits
                if current_state not in self.visits:
                    self.visits[current_state] = 1
                else:
                    self.visits[current_state] = self.visits[current_state] + 1
        
        # initialize distances with infinity for all elements
        self.state_dict.update({element: np.inf for element in self.distinct_elements})
        
        # set distance to goals as 0
        for goal in self.goals:
            self.state_dict[goal] = 0
        
        # initialize the priority queue (min-heap)
        self.pq = [(0, goal) for goal in self.goals]
        heapq.heapify(self.pq)
    
    def approximate_shortest_distance(self):
        
        print("[SDE] Nodes: ", len(self.graph))
        
        while self.pq:
            current_distance, current_state = heapq.heappop(self.pq)
            
            # skip already processed states
            if current_distance > self.state_dict[current_state]:
                continue
            
            # explore neighbors and update distances
            if current_state in self.graph:
                for next_state in self.graph[current_state]:
                    new_distance = current_distance + 1
                    if new_distance < self.state_dict[next_state]:
                        self.state_dict[next_state] = new_distance
                        heapq.heappush(self.pq, (new_distance, next_state))
                        if self.state_dict[next_state] != np.inf:
                            self.found_shortcuts += 1
    
    def get_shortest_distances(self):
        return self.state_dict
    
    def get_subgoals(self, num_subgoals, visit_threshold):
        
        visits_k = np.array(list(self.visits.keys()))
        visits_v = np.array(list(self.visits.values()))
        
        mask = visits_v > visit_threshold # don't consider very low visit states
        visits_k = visits_k[mask]
        visits_v = visits_v[mask]
        
        # sample based on inverse visits frequency
        total_visits = np.sum(visits_v)
        inverse_visits = total_visits / visits_v
        total_inverse_visits = np.sum(inverse_visits)

        probabilities = inverse_visits / total_inverse_visits
        subgoal_representations = np.random.choice(visits_k, num_subgoals, p=probabilities)
        return subgoal_representations
        
def join_txt(text): return np.asarray(",".join(text),dtype=object)