import numpy as np
import matplotlib.pyplot as plt
import os

# state variables
VAR_MAP = {
    "rho": 4,
    "u":   5,
    "v":   6,
    "p":   7
}


def plot_solution(folder, step, var="rho", ghost=3):
    """
    folder : output directory name
    step   : time step index (int)
    var    : 'rho', 'u', 'v', or 'p'
    ghost  : number of ghost cells
    """

    file_path = os.path.join(folder, f"{step}.npy")
    u = np.load(file_path)

    k = VAR_MAP[var]

    # remove ghost cells
    data = u[ghost:-ghost, ghost:-ghost, k]

    plt.figure(figsize=(6,5))
    plt.imshow(
        data.T,
        origin="lower",
        cmap="coolwarm",
        aspect="equal",
        vmin = 0,
        vmax = 1.2
    )
    print(data.T[5:,5:])
    plt.colorbar(label=var)
    plt.title(f"{var} at step {step}")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.tight_layout()
    plt.show()

#plot_solution("2026-01-21_14-38-34", step=15, var="rho")

import numpy as np
import matplotlib.pyplot as plt
import os

# state variables
VAR_MAP = {
    "rho": 4,
    "u":   5,
    "v":   6,
    "p":   7
}

def interactive_plot_keyboard(folder, initial_step=0, var="rho", ghost=3):
    """Interactive plot with keyboard navigation"""
    
    def get_available_steps(folder):
        steps = []
        for file in os.listdir(folder):
            if file.endswith('.npy'):
                try:
                    step = int(file.split('.')[0])
                    steps.append(step)
                except ValueError:
                    continue
        return sorted(steps)
    
    steps = get_available_steps(folder)
    if not steps:
        print(f"No .npy files found in {folder}")
        return
    
    current_step = initial_step if initial_step in steps else steps[0]
    current_var = var
    
    fig, ax = plt.subplots(figsize=(8, 6))
    img = None
    cbar = None
    
    def update_display(step, variable):
        nonlocal img, cbar
        
        file_path = os.path.join(folder, f"{step}.npy")
        u = np.load(file_path)
        k = VAR_MAP[variable]
        
        data = u[ghost:-ghost, ghost:-ghost, k]
        
        ax.clear()
        img = ax.imshow(
            data.T,
            origin="lower",
            cmap="coolwarm",
            aspect="equal",
        )
        
        if cbar:
            cbar.remove()
        cbar = fig.colorbar(img, ax=ax, label=variable)
        
        ax.set_title(f"{variable} at step {step}")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        
        plt.draw()
    
    def on_key(event):
        nonlocal current_step, current_var
        
        if event.key == 'right' or event.key == 'n':
            # Next step
            idx = steps.index(current_step)
            if idx < len(steps) - 1:
                current_step = steps[idx + 1]
                update_display(current_step, current_var)
                
        elif event.key == 'left' or event.key == 'p':
            # Previous step
            idx = steps.index(current_step)
            if idx > 0:
                current_step = steps[idx - 1]
                update_display(current_step, current_var)
                
        elif event.key == '1':
            # Change to rho
            current_var = 'rho'
            update_display(current_step, current_var)
        elif event.key == '2':
            # Change to u
            current_var = 'u'
            update_display(current_step, current_var)
        elif event.key == '3':
            # Change to v
            current_var = 'v'
            update_display(current_step, current_var)
        elif event.key == '4':
            # Change to p
            current_var = 'p'
            update_display(current_step, current_var)
    
    # Connect keyboard event
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display(current_step, current_var)
    
    # Add instructions
    print("Keyboard controls:")
    print("  Left arrow / 'p' : Previous step")
    print("  Right arrow / 'n': Next step")
    print("  '1' : Show density (rho)")
    print("  '2' : Show x-velocity (u)")
    print("  '3' : Show y-velocity (v)")
    print("  '4' : Show pressure (p)")
    
    plt.show()

#interactive_plot_keyboard("2026-02-11_00-24-45", initial_step=15, var="rho")
