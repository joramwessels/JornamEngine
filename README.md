# Path Tracer
## TODO
 - Test ray tracer  
	- query doesn't get initialized  
 - Test object loading  

## Log
***Week 1*** *11hrs*  
**23-04 09:00-11:00:** Planning  
**23-04 11:00-15:00:** Installing CUDA, Optix, and setting up VS environment  
**23-04 15:00-16:00:** Solving tutorial compiler errors  
**24-04 10:00-11:00:** Learning about Optix' call structure -- Optix and OpenGL are fully integrated, switch?  
**25-04 10:00-13:00:** Extracting Optix tutorial from SDK as template  
***Week 2*** *10hrs*  
**29-04 12:00-13:00:** Chose the SDL approach and switched everything back  
**30-04 09:00-11:00:** Reading Optix documentation and implementing main structure  
**30-04 13:00-15:00:** Reading Optix documentation and implementing scene loading  
**30-04 16:00-19:00:** Reading Optix documentation and implementing mesh loading  
**02-05 11:00-13:00:** Reading Optix documentation and implementing RTgeometryinstance  
***Week 3*** *11hrs*  
**06-05 11:30-14:30:** Switching to Optix Prime, object loading as RTPmodel  
**07-05 09:30-11:30:** Writing a transformation matrix struct for object placement  
**07-05 11:30-13:30:** Finishing scene parsing and scene loading; Preparing ray tracing  
**09-05 09:00-10:00:** Finishing ray tracer; The project compiles  
**09-05 10:00-13:00:** Debugging ray tracer  
***Week 4*** *4hrs*  
**16-05 10:00-12:00:** Debugging access violation exception  
**16-05 13:00-15:00:** Switching to Prime++, hoping to solve the exception  

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
 - Screen plotting straight from GPU

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
