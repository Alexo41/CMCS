import mesa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d
from agent import TumorCell

class TumorEnvironment:

    def __init__(self, width, height, init_oxygen = 1.0, init_glucose = 1.0):
        self.grid = mesa.space.SingleGrid(width, height, torus=False)
        self.width = width
        self.height = height

        # Initialize resource levels
        self.oxygen = np.full((width, height), init_oxygen, dtype=np.float32)
        self.glucose = np.full((width, height), init_glucose, dtype=np.float32)
        self.h_plus = np.zeros((width, height), dtype=np.float32)  

        # Visualization setup
        self.fig, self.ax = None, None
        self.im_cells, self.im_glucose, self.im_oxygen, self.im_h_plus = None, None, None, None

    def diffuse_resources(self, substeps=30):
        laplacian_kernel = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])
        # Coefficienti stabili — D_paper / substeps
        D_O2 = 1.04 / substeps
        D_G  = 5.24 / substeps
        D_H  = 0.63 / substeps

        for _ in range(substeps):
            delta = convolve2d(self.oxygen, laplacian_kernel,
                            mode='same', boundary='fill', fillvalue=1.0)
            self.oxygen += D_O2 * delta
            np.clip(self.oxygen, 0.0, None, out=self.oxygen)

            delta = convolve2d(self.glucose, laplacian_kernel,
                            mode='same', boundary='fill', fillvalue=1.0)
            self.glucose += D_G * delta
            np.clip(self.glucose, 0.0, None, out=self.glucose)

            delta = convolve2d(self.h_plus, laplacian_kernel,
                            mode='same', boundary='fill', fillvalue=0.0)
            self.h_plus += D_H * delta
            np.clip(self.h_plus, 0.0, None, out=self.h_plus)


    def supply_resources(self, borders_only=True):
        self.oxygen[0, :] = 1.0
        self.oxygen[-1, :] = 1.0
        self.oxygen[:, 0] = 1.0
        self.oxygen[:, -1] = 1.0

        self.glucose[0, :] = 1.0
        self.glucose[-1, :] = 1.0
        self.glucose[:, 0] = 1.0
        self.glucose[:, -1] = 1.0

        self.h_plus[0, :] = 0.0
        self.h_plus[-1, :] = 0.0
        self.h_plus[:, 0] = 0.0
        self.h_plus[:, -1] = 0.0
        
    def get_neighbourhood(self, pos):
        return self.grid.get_neighborhood(pos, moore=False, include_center=False, radius=1)
    
    def consume(self, pos, oxygen, glucose, h_plus):
        x, y = pos
        self.oxygen[x, y] = max(0.0, self.oxygen[x, y] - oxygen)
        self.glucose[x, y] = max(0.0, self.glucose[x, y] - glucose)  
        self.h_plus[x, y] += h_plus 

    def visualize(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 10))  # Adjusted for the new subplot
        self.ax = self.ax.flatten()  # Flatten to easily index subplots

        cmap_cells = colors.ListedColormap(['white', 'red', 'green', 'black', 'orange', 'blue'])

        self.im_cells = self.ax[0].imshow(np.zeros((self.width, self.height)), cmap=cmap_cells, vmin=0, vmax=5)
        self.ax[0].set_title("Cell States")

        self.im_oxygen = self.ax[1].imshow(self.oxygen, cmap='Greens', vmin=0, vmax=1)
        self.ax[1].set_title("Oxygen Levels")

        self.im_glucose = self.ax[2].imshow(self.glucose, cmap='Oranges', vmin=0, vmax=1)
        self.ax[2].set_title("Glucose Levels")

        self.im_h_plus = self.ax[3].imshow(self.h_plus, cmap='Blues', vmin=0, vmax=0.1)
        self.ax[3].set_title("H+ Levels")

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
        self.im_glucose.set_data(self.glucose)
        self.im_h_plus.set_data(self.h_plus)
        
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
        self.env.diffuse_resources()
        self.agents.shuffle_do("step")
        self.env.update_plot(self)

if __name__ == "__main__":
    model = TestModel(width=200, height=200)
    for i in range(200):
        print(f"Step {i+1}/200 running...")
        model.step()

    plt.ioff()
    plt.show()