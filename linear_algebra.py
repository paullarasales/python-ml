import numpy as np 
import matplotlib.pyplot as plt 

# Time range from 0 to 40 minutes with 4000 points
t = np.linspace(0, 40, 4000)

# Distance traveled by the robber (d_r) and the sheriff (d_s)
d_r = 2.5 * t 
d_s = 3 * (t - 5)

# Create the plot
fig, ax = plt.subplots()
plt.title('A Bank Robber Caught')
plt.xlabel('Time (in minutes)')
plt.ylabel('Distance (in kilometers)')
ax.set_xlim([0, 40])
ax.set_ylim([0, 100])
ax.plot(t, d_r, c='green', label='Robber')
ax.plot(t, d_s, c='brown', label='Sheriff')
plt.axvline(x=30, color='purple', linestyle='--')
plt.axhline(y=75, color='purple', linestyle='--')

# Add a legend
plt.legend()

# Show the plot
plt.show()

