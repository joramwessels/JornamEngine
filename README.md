# TODO
 - implement user input  
	- SDL key identifier doesn't match  
 - implement shading  
	- Color computation in generateShadowRay() returns 0
		- Try adding a separate color field for debugging
 - implement orthographic camera, DoF, FoV, and zoom  
 - port to GPU  

 # BUGS


 # OPTIMIZATIONS
 - screen buffer memcopies
 - rotation around world y axis
 - axis normalization after rotation
 - AVX instructions in Color functions (multiply - clamp)