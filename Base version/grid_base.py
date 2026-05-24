import mesa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d
from agent_base import TumorCell
import pandas as pd

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

    def diffuse_resources(self, substeps):
        # Simulates diffusion of oxygen through a Laplace discrete operator
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        
        D_O2 = 1.04 / substeps
        for _ in range(substeps):
            delta = convolve2d(self.oxygen, laplacian_kernel,
                            mode='same', boundary='fill', fillvalue=1.0)
            self.oxygen += D_O2 * delta
            np.clip(self.oxygen, 0.0, None, out=self.oxygen)

    def supply_resources(self):
        self.oxygen[0, :] = 1.0
        self.oxygen[-1, :] = 1.0
        self.oxygen[:, 0] = 1.0
        self.oxygen[:, -1] = 1.0
    
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
        self.history = []
        self.apoptosis_count = 0


        # Define centers
        center_x = width // 2
        center_y = height // 2
        
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
        self.env.diffuse_resources(substeps=50)
        self.agents.shuffle_do("step")

        counts = {
            'proliferating': 0,
            'quiescent': 0,
            'necrotic': 0,
            'apoptotic': 0,
        }

        for agent, _ in self.grid.coord_iter():
            if agent is not None:
                if agent.state == 1:
                    counts['proliferating'] += 1
                elif agent.state == 2:
                    counts['quiescent'] += 1
                elif agent.state == 3:
                    counts['necrotic'] += 1
        
        counts['apoptotic'] = self.apoptosis_count
        self.apoptosis_count = 0  # Reset for next step
        
        counts['alive'] = counts['proliferating'] + counts['quiescent']
        self.history.append(counts)
        
        self.env.update_plot(self)


if __name__ == "__main__":
    model = TestModel(width=300, height=300)
    for i in range(200):
        print(f"Step {i+1}/200 running...")
        model.step()

    plt.ioff()
    plt.show()

    
    # =====================================================================
    # PLOT MODELLO BASE (Da inserire nel file/cella del modello base)
    # =====================================================================
    df_base = pd.DataFrame(model.history)
    # 1. Dinamica della Popolazione
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_base['alive'],         label='Vive (Totali)',      color='deeppink', linewidth=2)
    ax.plot(df_base['necrotic'],      label='Necrotiche',         color='black',    linewidth=2)
    ax.plot(df_base['proliferating'], label='Proliferanti',       color='red',      linestyle='--')
    ax.plot(df_base['quiescent'],     label='Quiescenti',         color='green',    linestyle='--')
    ax.set_title("Modello Base: Dinamica Popolazione", fontsize=12, fontweight='bold')
    ax.set_xlabel("Step"); ax.set_ylabel("Numero di Cellule")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 2. Tasso di Apoptosi Istantanea
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_base['apoptotic'], color='purple', label='Morti per step', linewidth=1.5)
    ax.set_title("Modello Base: Apoptosi Istantanea", fontsize=12)
    ax.set_xlabel("Step"); ax.set_ylabel("Mortalità (Flusso)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()