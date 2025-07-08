import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import TwoSlopeNorm
import math

from simulator.random_state_generator import RandomStateGenerator
from simulator.descent import State, calculate_inital_guess, calculate_error

N = 50
az = np.linspace(-np.pi, np.pi, N)
el = np.linspace(-np.pi, np.pi, N)
AZ, EL = np.meshgrid(az, el)

state = {
    'gun_name': 'Battle Cannon',
    'muzzle_velocity': 800,
    'projectile_drag': 0.002,
    'gravity': 0.5,
    'heading': 59.29,
    'target_velocity_heading': 90,
    'my_velocity_heading': 134.25,
    'target_distance': 710.75,
    'target_altitude': 5.95,
    'target_velocity': 1000,
    'my_velocity': -52.07,
    'wind_velocity': 0.28,
    'wind_direction': 311.86,
    't_x': 362.92,
    't_y': 5.95,
    't_z': 611.12,
    'target': (362.92, 5.95, 611.12),
}

state = {'gun_name': 'Battle Cannon', 'muzzle_velocity': 800, 'projectile_drag': 0.002, 'gravity': 0.5, 'heading': 59.2956752826775, 'target_velocity_heading': 230.84666723026086, 'my_velocity_heading': 134.25571905690776, 'target_distance': 710.753897576432, 'target_altitude': 5.949174098593654, 'target_velocity': 9.272491793836092, 'my_velocity': -52.0658262415362, 'wind_velocity': 0.2801319412540215, 'wind_direction': 311.86513867406103, 't_x': 362.91649719239274, 't_y': 5.949174098593654, 't_z': 611.1159619791429, 'target': (362.91649719239274, 5.949174098593654, 611.1159619791429)}
# state = {'gun_name': 'Artillery Cannon', 'muzzle_velocity': 700, 'projectile_drag': 0.001, 'gravity': 0.5, 'heading': 92.45931633092162, 'target_velocity_heading': 54.347145866697645, 'my_velocity_heading': 21.721086933148854, 'target_distance': 1188.7246619494563, 'target_altitude': 20.980773977537687, 'target_velocity': 8.35029010366333, 'my_velocity': 42.93872937895494, 'wind_velocity': 0.32846046978016297, 'wind_direction': 198.2978020017737, 't_x': -51.008161306524876, 't_y': 20.980773977537687, 't_z': 1187.6297779219653, 'target': (-51.008161306524876, 20.980773977537687, 1187.6297779219653)}
# state = {'gun_name': 'Bertha Cannon', 'muzzle_velocity': 600, 'projectile_drag': 0.0005, 'gravity': 0.5, 'heading': 355.67018100650796, 'target_velocity_heading': 7.929385840150456, 'my_velocity_heading': 71.35381019016533, 'target_distance': 596.4727536739665, 'target_altitude': 54.72554208979237, 'target_velocity': 52.33670825529765, 'my_velocity': 28.795554930636, 'wind_velocity': 0.8162675669867526, 'wind_direction': 74.97734279995188, 't_x': 594.7704064653472, 't_y': 54.72554208979237, 't_z': -45.0323158237507, 'target': (594.7704064653472, 54.72554208979237, -45.0323158237507)}

gradient_state = State(state)
guess_az, guess_el, guess_time = calculate_inital_guess(state)
print(guess_az, guess_el, guess_time)
z_init = guess_time

# Plot setup
fig, ax = plt.subplots(figsize=(15, 14))
plt.subplots_adjust(bottom=0.25)

def compute_field(z):
    U = -np.vectorize(gradient_state.partial_az)(AZ, EL, z)
    V = -np.vectorize(gradient_state.partial_el)(AZ, EL, z)
    mag = np.sqrt(U**2 + V**2)
    T = -np.vectorize(gradient_state.partial_time)(AZ, EL, z)
    return U, V, mag, T

U, V, mag, T = compute_field(z_init)

rects = ax.imshow(T, extent=(az[0], az[-1], el[0], el[-1]), origin='lower', cmap='coolwarm', aspect='auto', alpha=0.8)
quiver = ax.quiver(AZ, EL, U, V, mag, cmap='plasma')

prediction_dot, = ax.plot(guess_az, guess_el, 'wo', markersize=8, markeredgecolor='black', label='Prediction')

plt.colorbar(rects, ax=ax, fraction=0.046, pad=0.04).set_label('∂f/∂t')
plt.colorbar(quiver, ax=ax, fraction=0.046, pad=0.08).set_label('Gradient Magnitude')

ax.set_title(f"Gradient Vector Field with Time Gradient at t = {z_init:.2f}")
ax.set_xlabel("azimuth")
ax.set_ylabel("elevation")
ax.axis('equal')
ax.grid(True)

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 't', 0, guess_time + 60, valinit=z_init)

def update_visual(z, az_val, el_val):
    U, V, mag, T = compute_field(z)
    quiver.set_UVC(U, V, mag)

    if np.all(T >= 0):
        rects.set_cmap('Reds')
        rects.set_norm(plt.Normalize(vmin=np.min(T), vmax=np.max(T)))
        rects.set_data(T)
    elif np.all(T <= 0):
        rects.set_cmap('Blues')
        T_mapped = -T
        rects.set_norm(plt.Normalize(vmin=np.min(T_mapped), vmax=np.max(T_mapped)))
        rects.set_data(T_mapped)
    else:
        rects.set_cmap('coolwarm')
        rects.set_norm(TwoSlopeNorm(vcenter=0, vmin=np.min(T), vmax=np.max(T)))
        rects.set_data(T)

    prediction_dot.set_data(az_val, el_val)
    ax.set_title(f"Gradient Descent at t = {z:.2f}")
    slider.set_val(z)
    plt.pause(0.001)

# Generator version of gradient descent
def gradient_descent_live(firing_state, learning_rate=0.00003, tolerance=0.1, max_iterations=5000):
    current_state = State(firing_state)
    current_guess = calculate_inital_guess(firing_state)
    error = float('inf')
    count = 0

    error_increased_count = 0
    flipped_time_gradient = False
    time_gradient_scalar = 1

    while error > tolerance and count < max_iterations:
        count += 1

        gradient_az   = 0.01 * current_state.partial_az(*current_guess)
        gradient_el   = 0.01 * current_state.partial_el(*current_guess)
        gradient_time = current_state.partial_time(*current_guess)

        new_az   = current_guess[0] - learning_rate * gradient_az
        new_el   = current_guess[1] - learning_rate * gradient_el
        new_time = current_guess[2] - learning_rate * gradient_time * time_gradient_scalar

        current_guess = (new_az, new_el, new_time)
        new_error = calculate_error(firing_state, current_guess)

        if not flipped_time_gradient and new_error > error:
            error_increased_count += 1

            if error_increased_count > 10:
                print('Increasing error detected, flipped time gradient')
                flipped_time_gradient = True
                time_gradient_scalar = -1
        else:
            error_increased_count = 0

        error = new_error
        print(error)

        yield current_guess  # for visual update

# Animate the descent
for az_val, el_val, t_val in gradient_descent_live(state):
    update_visual(t_val, az_val, el_val)
