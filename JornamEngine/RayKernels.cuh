#pragma once

#ifdef __CUDACC__
//
extern void createPrimaryRaysOnDevice(float3* rays, unsigned int width, unsigned int height,
	JornamEngine::Camera* camera, unsigned int blockX = 32, unsigned int blockY = 16);

#endif