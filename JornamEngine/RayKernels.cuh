#pragma once

//
extern void createPrimaryRaysOnDevice(float3* rays, unsigned int width, unsigned int height,
	float3 eye, float3 TR, float3 TL, float3 BL, unsigned int blockX = 32, unsigned int blockY = 16);