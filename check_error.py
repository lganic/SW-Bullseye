import numpy as np
import matplotlib.pyplot as plt
from simulator.descent import calculate_error, calculate_inital_guess

state = {'gun_name': 'Battle Cannon', 'muzzle_velocity': 800, 'projectile_drag': 0.002, 'gravity': 0.5, 'heading': 59.2956752826775, 'target_velocity_heading': 230.84666723026086, 'my_velocity_heading': 134.25571905690776, 'target_distance': 710.753897576432, 'target_altitude': 5.949174098593654, 'target_velocity': -9.272491793836092, 'my_velocity': -52.0658262415362, 'wind_velocity': 0.2801319412540215, 'wind_direction': 311.86513867406103, 't_x': 362.91649719239274, 't_y': 5.949174098593654, 't_z': 611.1159619791429, 'target': (362.91649719239274, 5.949174098593654, 611.1159619791429)}
# state = {'gun_name': 'Artillery Cannon', 'muzzle_velocity': 700, 'projectile_drag': 0.001, 'gravity': 0.5, 'heading': 92.45931633092162, 'target_velocity_heading': 54.347145866697645, 'my_velocity_heading': 21.721086933148854, 'target_distance': 1188.7246619494563, 'target_altitude': 20.980773977537687, 'target_velocity': -8.35029010366333, 'my_velocity': 42.93872937895494, 'wind_velocity': 0.32846046978016297, 'wind_direction': 198.2978020017737, 't_x': -51.008161306524876, 't_y': 20.980773977537687, 't_z': 1187.6297779219653, 'target': (-51.008161306524876, 20.980773977537687, 1187.6297779219653)}
state = {'gun_name': 'Bertha Cannon', 'muzzle_velocity': 600, 'projectile_drag': 0.0005, 'gravity': 0.5, 'heading': 355.67018100650796, 'target_velocity_heading': 7.929385840150456, 'my_velocity_heading': 71.35381019016533, 'target_distance': 596.4727536739665, 'target_altitude': 54.72554208979237, 'target_velocity': 52.33670825529765, 'my_velocity': 28.795554930636, 'wind_velocity': 0.8162675669867526, 'wind_direction': 74.97734279995188, 't_x': 594.7704064653472, 't_y': 54.72554208979237, 't_z': -45.0323158237507, 'target': (594.7704064653472, 54.72554208979237, -45.0323158237507)}

# Get the az, el from initial guess
guess_az, guess_el, guess_time = calculate_inital_guess(state)

# Time range to sweep
t_vals = np.linspace(0, 2 * guess_time, 300)

# Calculate error at each t
errors = [calculate_error(state, (guess_az, guess_el, t)) for t in t_vals]

# Plot
plt.figure(figsize=(8, 4))
plt.plot(t_vals, errors, label='Error vs Time')
plt.axvline(x=guess_time, color='gray', linestyle='--', label='Initial guess')
plt.xlabel('Time (t)')
plt.ylabel('Squared Distance Error')
plt.title('Error vs Time at Fixed Az/El')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
