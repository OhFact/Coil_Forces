# Coil Forces
This code was written in order to study the forces and magnetic fields generated inside and between stellarator coils, in order to be able to run simulations of coil displacement.

### Executing program
First, create a "results" and a "results_txt" folder to run this code  
  
Default inputs are 150000 Amperes of current running through the HSX coils in this repo, alongside a rectangular cross section 0.3 x 0.06 m  
  
hsx_utilities > Functions for loading file geometry, running the force calculations, plotting, etc.  
fig_reproduce > Reproduces all the plots of the original paper this code is based on  
tpq_frame > prints out forces in a .txt file in the tpq frame described in the paper
animate_B > Animates the magnetic field in the coils as we go around the coils length  
animate_forces > Animates the forces on the coil as we go around the coils length
  
This code is based on this paper, alongside advice from Mark Nornberg and Benedikt Geiger at HSX, UW-Madison:  
Efficient calculation of self magnetic field, self-force, and self-inductance for electromagnetic coils. II. Rectangular cross-section, Landreman et al.

### Dependencies  
Packages required can be seen below:  
numpy>=2.0.0  
scipy>=1.15.0  
jax>=0.9.0  
jaxlib>=0.9.0  
matplotlib>=3.9.0

