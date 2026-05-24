import mesa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d
from agent import TumorCell
import pandas as pd

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

    def diffuse_resources(self, substeps=50):
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


    def supply_resources(self):
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
        self.env.diffuse_resources()
        self.agents.shuffle_do("step")
        
        counts = {
            'prolif_aero':   0,
            'prolif_glyco':  0,
            'quiesc_aero':   0,
            'quiesc_glyco':  0,
            'necrotic':      0,
            'apoptotic':     self.apoptosis_count
        }
        for agent, _ in self.grid.coord_iter():
            if agent is not None:
                if agent.state == 1:
                    counts['prolif_aero'] += 1
                elif agent.state == 2:
                    counts['quiesc_aero'] += 1
                elif agent.state == 3:
                    counts['necrotic'] += 1
                elif agent.state == 4:
                    counts['prolif_glyco'] += 1
                elif agent.state == 5:
                    counts['quiesc_glyco'] += 1

        counts['alive_aero'] = counts['prolif_aero'] + counts['quiesc_aero']
        counts['alive_glyco'] = counts['prolif_glyco'] + counts['quiesc_glyco']
        counts['alive'] = counts['alive_aero'] + counts['alive_glyco']
        counts['apoptotic'] = self.apoptosis_count
        self.apoptosis_count = 0  # Reset for next step
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
    # PLOT MODELLO WARBURG (Da inserire nel file/cella del modello Warburg)
    # =====================================================================
    df_warb = pd.DataFrame(model.history)
    # 1. Dinamica della Popolazione
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_warb['alive'],         label='Vive (Totali)',      color='deeppink', linewidth=2)
    ax.plot(df_warb['alive_glyco'],   label='Vive (Glicolitiche)',color='orange',   linewidth=2)
    ax.plot(df_warb['necrotic'],      label='Necrotiche',         color='black',    linewidth=2)
    ax.plot(df_warb['prolif_aero'],   label='Proliferanti (Aero)',color='red',      linestyle='--')
    ax.plot(df_warb['prolif_glyco'],  label='Proliferanti (Glico)',color='darkgoldenrod', linestyle='-.')
    ax.plot(df_warb['quiesc_aero'],   label='Quiescenti (Aero)',  color='green',    linestyle='--')
    ax.plot(df_warb['quiesc_glyco'],  label='Quiescenti (Glico)', color='blue',     linestyle='-.')
    ax.set_title("Modello Warburg: Dinamica Popolazione", fontsize=12, fontweight='bold')
    ax.set_xlabel("Step"); ax.set_ylabel("Numero di Cellule")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 2. Tasso di Apoptosi Istantanea
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_warb['apoptotic'], color='purple', label='Morti per step', linewidth=1.5)
    ax.set_title("Modello Warburg: Apoptosi Istantanea", fontsize=12)
    ax.set_xlabel("Step"); ax.set_ylabel("Mortalità (Flusso)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 3. Frazione Glicolitica
    fig, ax = plt.subplots(figsize=(8, 4))
    # Calcolo con np.nan per evitare errori di divisione per zero nei primi step
    glyco_frac = df_warb['alive_glyco'] / df_warb['alive'].replace(0, np.nan)
    ax.plot(glyco_frac, color='darkgoldenrod', label='Frazione Glicolitica', linewidth=2.5)
    ax.set_title("Modello Warburg: Frazione Cellule Glicolitiche", fontsize=12, fontweight='bold')
    ax.set_xlabel("Step"); ax.set_ylabel("Frazione (0.0 - 1.0)")
    ax.set_ylim(-0.05, 1.05) # Mantiene il grafico stabile bloccando la percentuale tra 0 e 100%
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()