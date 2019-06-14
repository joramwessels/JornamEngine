#pragma once


/*
	Initializes the primary rays on the GPU

	@param rays		A device pointer to an array of consecutive (origin, direction) pairs
	@param width	The width of the screen in pixels
	@param height	The height of the screen in pixels
*/
extern void createPrimaryRaysOnDevice(float3* rays, unsigned int width, unsigned int height,
	float3 eye, float3 TR, float3 TL, float3 BL, unsigned int blockX = 32, unsigned int blockY = 16);