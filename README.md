# Path Tracer
## TODO
 - Write simple renderer to test Optix
 - Create scene in Optix environment (GeometryGroup)  

## Log
***Week 1***  
**23-04 09:00-:** Planning & NVidia Optix import  

## Planning
***Week 1:*** Optix tutorial  
**23-04:** Plan first two weeks, work on normal shader  
**25-04:** Get normal shader up and running with existing framework  
**27-04:** Buffer
***Week 2:*** Simple shader  
**29-04:** Extend normal shader to diffuse shader  
**30-04:** Finish diffuse shader  
**02-05:** Plan path tracer implementation
***Week 3:*** Raw Path Tracer  
***Week 4:*** Stratification, NEE & MIS  
***Week 5:*** Stratification, NEE & MIS  

## Backlog
 - Raw path tracer  
 - Stratification  
 - NEE  
 - MIS  
 - BVH  
 - Mesh management  
 - Texture management  
 - Blinn-Phong BRDFs  
 - User interface  
 - Normal mapping  
 - DOF, FoV & zoom  

## Bugs
 -

\
\
\
\

# Engine
## TODO
 - implement user input
	- SDL key identifier doesn't match

## Backlog
 - Graphics  
	- Path tracer  
	- Object loading  
	- Texture mapping  
	- Normal mapping  
 - UI  
	- Looking  
	- Walking  
	- Zoom  
	- Picking up documents  
 - Physics  
	- Player gravity & collisions  
	- Stairs  
	- Wind  
	- Waving water surfaces  
 - Sound  
 - Scene editor  

## Bugs
 - 

## Potential Optimizations
 - screen buffer memcopies
 - rotation around world y axis
 - axis normalization after rotation
 - AVX instructions in Color functions (multiply - clamp)
 -