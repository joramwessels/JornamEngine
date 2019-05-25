# Path Tracer
## TODO
- Diffuse shading  
- Enable UI  
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
***Week 4*** *8hrs*  
**16-05 10:00-12:00:** Debugging access violation exception  
**16-05 13:00-15:00:** Switching to Prime++, hoping to solve the exception  
**17-05 10:00-11:00:** New code requires cuda.h - issues with importing includes  
**18-05 12:00-13:00:** Code builds - solving Optix Prime exceptions  
**19-05 10:30-12:30:** Triangle model update throws "invalid value" exception  
***Week 5*** *9hrs*  
**20-05 12:00-15:00:** Incorporated simple linear example code - pipeline works on CPU  
**23-05 10:00-12:00:** Locating the difference between example code and original code: model handles  
**23-05 14:00-15:00:** Experimented with model passing  
**25-05 10:30-13:30:** Fixed scene loading; changed Model struct to include normals; new planning  

## Planning
***Week 6:*** User interface  
***Week 7:*** Raw path tracer w/ area lights  
***Week 8:*** NEE and Importance Sampling  
***Week 9:*** MIS and Blinn-Phong shading  
***Week 10:*** Textures  

## Backlog
 - Raw path tracer  
 - Stratification  
 - NEE  
 - MIS  
 - ~~BVH~~  
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
