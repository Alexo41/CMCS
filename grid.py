import mesa
from networkx import radius
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation

class TumorEnvironment:

    def __init__(self, width, height, init_glucose = 1.0, init_oxygen = 1.0):
        self.grid = mesa.space.SingleGrid(width, height, torus=False)
        self.width = width
        self.height = height

        # Initialize glucose and oxygen levels
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y, indexing='ij')

        distance = np.sqrt(X**2 + Y**2)
        gradient = np.clip(distance, 0.1, 1.0)  # Avoid zero to prevent division issues
        
        self.glucose = np.copy(gradient)
        self.oxygen = np.copy(gradient)

        # Visualization setup
        self.fig, self.ax = None, None
        self.im_cells, self.im_glucose, self.im_oxygen = None, None, None

    def diffuse_resources(self, diffusion_rate_g, diffusion_rate_o):
        # TODO: vedere equazioni di diffusione
        pass

    def supply_resources(self, borders_only=True):
        # TODO: vedere meglio
        pass
        
    def get_neighbourhood(self, pos):
        return self.grid.get_neighborhood(pos, moore=True, include_center=False, radius=1)
    
    def consume(self, pos, glucose, oxygen):
        x, y = pos
        self.glucose[x, y] = max(0.0, self.glucose[x, y] - glucose)
        self.oxygen[x, y] = max(0.0, self.oxygen[x, y] - oxygen)

    def visualize(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))

        cmap_cells = colors.ListedColormap(['white', 'red', 'black'])

        self.im_cells = self.ax[0].imshow(np.zeros((self.width, self.height)), cmap=cmap_cells, vmin=0, vmax=3)
        self.ax[0].set_title("Cell States")

        self.im_glucose = self.ax[1].imshow(self.glucose, cmap='Blues', vmin=0, vmax=1)
        self.ax[1].set_title("Glucose Levels")

        self.im_oxygen = self.ax[2].imshow(self.oxygen, cmap='Greens', vmin=0, vmax=1)
        self.ax[2].set_title("Oxygen Levels")

        plt.tight_layout()

    def update_plot(self, model):
        if self.fig is None:
            self.visualize()

        agent_matrix = np.zeros((self.width, self.height))
        for cell, (x, y) in model.grid.coord_iter():
            if cell:
                agent_matrix[x, y] = cell.state  # Assuming one agent per cell
        
        self.im_cells.set_data(agent_matrix)
        self.im_glucose.set_data(self.glucose)
        self.im_oxygen.set_data(self.oxygen)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  
        plt.pause(0.01)  # Small pause to update the plot


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

        self.model.env.consume(self.pos, glucose=0.1, oxygen=0.05)  # Consume resources


class TestModel(mesa.Model):
    def __init__(self, width, height, n_agents):
        super().__init__()
        self.env = TumorEnvironment(width, height)
        self.grid = self.env.grid

        for i in range(n_agents):
            a = DummyWalker(self)

            while True:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                if self.env.grid.is_cell_empty((x, y)):
                    self.env.grid.place_agent(a, (x, y))
                    break
            
    def step(self):
        self.agents.shuffle_do("step")
        self.env.update_plot(self)


if __name__ == "__main__":
    model = TestModel(width=50, height=50, n_agents=20)
    for i in range(50):
        model.step()

    plt.ioff()
    plt.show()