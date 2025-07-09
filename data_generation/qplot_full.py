import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import TwoSlopeNorm

from simulator.random_state_generator import RandomStateGenerator
from simulator.descent import State, calculate_inital_guess

N = 50

az = np.linspace(-np.pi, np.pi, N)
el = np.linspace(-np.pi, np.pi, N)
AZ, EL = np.meshgrid(az, el)

state = {'gun_name': 'Battle Cannon', 'muzzle_velocity': 800, 'projectile_drag': 0.002, 'gravity': 0.5, 'heading': 59.2956752826775, 'target_velocity_heading': 230.84666723026086, 'my_velocity_heading': 134.25571905690776, 'target_distance': 710.753897576432, 'target_altitude': 5.949174098593654, 'target_velocity': -9.272491793836092, 'my_velocity': -52.0658262415362, 'wind_velocity': 0.2801319412540215, 'wind_direction': 311.86513867406103, 't_x': 362.91649719239274, 't_y': 5.949174098593654, 't_z': 611.1159619791429, 'target': (362.91649719239274, 5.949174098593654, 611.1159619791429)}

state = {'gun_name': 'Battle Cannon', 'muzzle_velocity': 800, 'projectile_drag': 0.002, 'gravity': 0.5, 'heading': 59.2956752826775, 'target_velocity_heading': 230.84666723026086, 'my_velocity_heading': 134.25571905690776, 'target_distance': 710.753897576432, 'target_altitude': 5.949174098593654, 'target_velocity': -9.272491793836092, 'my_velocity': -52.0658262415362, 'wind_velocity': 0.2801319412540215, 'wind_direction': 311.86513867406103, 't_x': 362.91649719239274, 't_y': 5.949174098593654, 't_z': 611.1159619791429, 'target': (362.91649719239274, 5.949174098593654, 611.1159619791429)}

state['target_velocity'] = 1000
state['target_velocity_heading'] = 90

gradient_state = State(state)

guess_az, guess_el, guess_time = calculate_inital_guess(state)
z_init = guess_time

# Set up figure with one axis
fig, ax = plt.subplots(figsize=(7, 6))
plt.subplots_adjust(bottom=0.25)

def compute_field(z):
    # Negative gradient to show gradient descent direction

    U = -np.vectorize(gradient_state.partial_az)(AZ, EL, z)
    V = -np.vectorize(gradient_state.partial_el)(AZ, EL, z)
    mag = np.sqrt(U**2 + V**2)
    T = -np.vectorize(gradient_state.partial_time)(AZ, EL, z)
    return U, V, mag, T

# Initial field
U, V, mag, T = compute_field(z_init)

# Plot heatmap first
rects = ax.imshow(T, extent=(az[0], az[-1], el[0], el[-1]),
                  origin='lower', cmap='coolwarm', aspect='auto', alpha=0.8)

# Overlay quiver plot
quiver = ax.quiver(AZ, EL, U, V, mag, cmap='plasma')

prediction_dot, = ax.plot(guess_az, guess_el, 'wo', markersize=8, markeredgecolor='black', label='Prediction')

# Colorbars
cbar1 = plt.colorbar(rects, ax=ax, fraction=0.046, pad=0.04)
cbar1.set_label('∂f/∂t')

cbar2 = plt.colorbar(quiver, ax=ax, fraction=0.046, pad=0.08)
cbar2.set_label('Gradient Magnitude')

ax.set_title(f"Gradient Vector Field with Time Gradient at t = {z_init:.2f}")
ax.set_xlabel("azimuth")
ax.set_ylabel("elevation")
ax.axis('equal')
ax.grid(True)

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 't', 0, guess_time + 60, valinit=z_init)

# Update function
def update(val):
    z = slider.val
    U, V, mag, T = compute_field(z)
    quiver.set_UVC(U, V, mag)

    # Determine how to normalize and color
    if np.all(T >= 0):
        # Entirely positive: use Reds colormap
        rects.set_cmap('Reds')
        rects.set_norm(plt.Normalize(vmin=np.min(T), vmax=np.max(T)))
        rects.set_data(T)
    elif np.all(T <= 0):
        # Entirely negative: use Blues colormap
        rects.set_cmap('Blues')
        T_mapped = -T
        rects.set_norm(plt.Normalize(vmin=np.min(T_mapped), vmax=np.max(T_mapped)))
        rects.set_data(T_mapped)
    else:
        # Mixed: diverging map centered at 0
        rects.set_cmap('coolwarm')
        rects.set_norm(TwoSlopeNorm(vcenter=0, vmin=np.min(T), vmax=np.max(T)))
        rects.set_data(T)
    
    prediction_dot, = ax.plot(guess_az, guess_el, 'wo', markersize=8, markeredgecolor='black', label='Prediction')

    ax.set_title(f"Gradient Vector Field with Time Gradient at t = {z:.2f}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
