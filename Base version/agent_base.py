import mesa
import numpy as np


class DummyWalker(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.state = 1

    def step(self):
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty_neighbors = [n for n in neighbors if self.model.grid.is_cell_empty(n)]
        
        if empty_neighbors:
            new_pos = self.random.choice(empty_neighbors)
            self.model.env.grid.move_agent(self, new_pos)

        self.model.env.consume(self.pos, oxygen=0.05)  # Consume resources


class TumorCell(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.state = 1  
        self.hidden_w = np.array([
            [ 1.0,   0.0],   
            [ 0.5,   0.0],   
            [ 0.0,  -2.0],   
            [ 1.0,   0.0],    
            [ 0.0,   0.0]    # Additional hidden neuron for future ph output
        ], dtype=np.float32)
        self.theta_initial = np.array([0.55, 0.0, 0.7, -0.25, 0.0], dtype=np.float32)
        self.output_w = np.array([
            [-0.5,   1.0,  -0.5,   0.0,   0.0],  # Pesi verso Proliferazione
            [ 0.0,   0.55, -0.5,   0.0,   0.0],  # Pesi verso Quiescenza
            [ 0.0,   0.0,   2.0,   -2.0,   0.0]   # Pesi verso Apoptosi
        ], dtype=np.float32)  
        self.phi_initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-2 *x))
    
    def _mutate(self, w):
        flat = w.flatten()

        n_mutations = np.random.poisson(0.01 * len(flat))

        if n_mutations > 0:
            indices = np.random.choice(len(flat), size=min(n_mutations, len(flat)), replace=False)
            flat[indices] += np.random.normal(0, 0.25, size=len(indices))
        
        return flat.reshape(w.shape)
    
    
    def proliferation(self, neighbors):
        # Place child in a random empty neighboring cell
        empty_neighbors = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]

        if not empty_neighbors:
            self.quiescence()  # No space to proliferate, switch to quiescence
            return 
        else:
            # Change state and consume oxygen
            self.state = 1  # Proliferating state
            
            # Copy parent's parameters
            new_hidden_w = self.hidden_w.copy()
            new_theta_initial = self.theta_initial.copy()
            new_output_w = self.output_w.copy()
            new_phi_initial = self.phi_initial.copy()

            # Apply mutation
            child = TumorCell(self.model)
            child.hidden_w = self._mutate(new_hidden_w)
            child.theta_initial = self._mutate(new_theta_initial)
            child.output_w = self._mutate(new_output_w)
            child.phi_initial = self._mutate(new_phi_initial)

            # Place child in a random empty neighboring cell
            child_pos = self.random.choice(empty_neighbors)
            self.model.grid.place_agent(child, child_pos)

    def quiescence(self):
        self.state = 2 # Quiescent state

    def apoptosis(self):
        self.model.grid.remove_agent(self)
        self.model.agents.discard(self)

    def step(self):
        # Get inputs for the neural network computing the new state
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        num_neighbors = 0
        for n in neighbors:
            if not self.model.grid.is_cell_empty(n):
                num_neighbors += 1
        
        x, y = self.pos        
        oxygen_level = self.model.env.oxygen[x, y]

        # Feed-forward computation
        input = np.array([num_neighbors / 4.0, oxygen_level])
        hidden = self._sigmoid(np.dot(self.hidden_w, input) - self.theta_initial)
        output = self._sigmoid(np.dot(self.output_w, hidden) - self.phi_initial)

        # Highest output value determines the action
        # Proliferation = 0, Quiescence = 1, Apoptosis = 2
        action = np.argmax(output)

        # Determine resource consumption based on action
        if action == 0:
            rate = 0.0667 
        elif action == 1:
            rate = 0.0133
        else:
            rate = 0.0

        # Necrosis case
        if rate > 0 and self.model.env.oxygen[x, y] < rate:
            # Not enough oxygen to perform the action, switch to quiescence
            self.state = 3  # Necrotic state
            self.model.agents.discard(self)  
            return

        # Consume oxygen
        if rate > 0:
            self.model.env.consume(self.pos, oxygen=rate)  
            
        # Execute action and consume resources accordingly
        if action == 0:
            self.proliferation(neighbors)
        elif action == 1:
            self.quiescence()
        else:
            self.apoptosis()
        
        
       