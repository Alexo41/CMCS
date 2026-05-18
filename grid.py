import mesa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d
from agent import TumorCell, DummyWalker

class TumorEnvironment:

    def __init__(self, width, height, init_oxygen = 1.0):
        self.grid = mesa.space.SingleGrid(width, height, torus=False)
        self.width = width
        self.height = height

        # Initialize oxygen level
        self.oxygen = np.full((width, height), init_oxygen, dtype=np.float32)

        # Visualization setup
        self.fig, self.ax = None, None
        self.im_cells, self.im_glucose, self.im_oxygen = None, None, None

    def diffuse_resources(self, diff_c = 0.1):
        # Simulates diffusion of oxygen through a Laplace discrete operator
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        
        delta_oxygen = convolve2d(self.oxygen, laplacian_kernel, mode='same', boundary='fill', fillvalue=1.0)
        self.oxygen += diff_c * delta_oxygen
        # Assures that oxygen levels do not go negative
        np.clip(self.oxygen, 0.0, None, out=self.oxygen) 

    def supply_resources(self, borders_only=True):
        self.oxygen[0, :] = 1.0
        self.oxygen[-1, :] = 1.0
        self.oxygen[:, 0] = 1.0
        self.oxygen[:, -1] = 1.0
        
    def get_neighbourhood(self, pos):
        return self.grid.get_neighborhood(pos, moore=False, include_center=False, radius=1)
    
    def consume(self, pos, oxygen):
        x, y = pos
        self.oxygen[x, y] = max(0.0, self.oxygen[x, y] - oxygen)

    def visualize(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))

        cmap_cells = colors.ListedColormap(['white', 'red', 'green', 'black'])

        self.im_cells = self.ax[0].imshow(np.zeros((self.width, self.height)), cmap=cmap_cells, vmin=0, vmax=3)
        self.ax[0].set_title("Cell States")

        self.im_oxygen = self.ax[1].imshow(self.oxygen, cmap='Greens', vmin=0, vmax=1)
        self.ax[1].set_title("Oxygen Levels")

        plt.tight_layout()

    def update_plot(self, model):
        if self.fig is None:
            self.visualize()

        agent_matrix = np.zeros((self.width, self.height))
        for cell, (x, y) in model.grid.coord_iter():
            if cell:
                agent_matrix[x, y] = cell.state  # Assuming one agent per cell
        
        self.im_cells.set_data(agent_matrix)
        self.im_oxygen.set_data(self.oxygen)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  
        plt.pause(0.1)  # Small pause to update the plot


class TestModel(mesa.Model):
    def __init__(self, width, height):
        super().__init__()
        self.env = TumorEnvironment(width, height)
        self.grid = self.env.grid

        # Define centers
        center_x = width // 2
        center_y = height // 2

        """ 
        # Cross shape of initial tumor cells
        initial_positions = [
            (center_x, center_y), 
            (center_x + 1, center_y),
            (center_x - 1, center_y), 
            (center_x, center_y + 1), 
            (center_x, center_y - 1)]
        
        # Single initial tumor cell
        initial_positions = [(center_x, center_y)]
        """
        
        # 2x2 block of initial tumor cells
        initial_positions = [
            (center_x, center_y), 
            (center_x + 1, center_y), 
            (center_x, center_y + 1), 
            (center_x + 1, center_y + 1)]
        
        
        for pos in initial_positions:
            a = TumorCell(self)
            self.grid.place_agent(a, pos)

            
    def step(self):
        self.env.supply_resources()
        self.env.diffuse_resources(diff_c=0.1)
        self.agents.shuffle_do("step")
        self.env.update_plot(self)

if __name__ == "__main__":
    model = TestModel(width=100, height=100)
    for i in range(100):
        print(f"Step {i+1}/100 running...")
        model.step()

    plt.ioff()
    plt.show()