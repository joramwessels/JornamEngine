#pragma once

#ifndef JORNAMENGINE
#define JORNAMENGINE

// Disbling deprecation warnings for buffer functions in stdio
#define _CRT_SECURE_NO_DEPRECATE

// Enabling debug mode (define JE_DEBUG in preprocessor to enable)
#ifdef JE_DEBUG
#define JE_DEBUG_MODE true
#else
#define JE_DEBUG_MODE false
#endif

// Windows
#ifdef _WIN32
#include <Windows.h>
#endif

// External dependencies
#include <SDL.h>
#include <FreeImage.h>
#include "tiny_obj_loader.h"

// console
#include <fcntl.h>
#include <io.h>


// C++ headers
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <vector>

// C headers
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <conio.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>

// AVX intrinsics
#include <immintrin.h>

// CUDA
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <curand.h>
//#include <curand_kernel.h>

// Optix
//#define NOMINMAX

//#include <Optix/optixu/optixpp_namespace.h>
//#include <Optix/optixu/optixu_math_stream_namespace.h>
#include <Optix/optix.h>
//#include <Optix/optixu/optixu_math_namespace.h>
//#include <Optix/optix_math.h>
//#define RADIANCE_RAY_TYPE 0
//#define SHADOW_RAY_TYPE 1
//#include <optix/optixu/optixu_vector_types.h>
#include <Optix/optix_prime/optix_prime.h>
#include <Optix/optix_prime/optix_primepp.h>

//#include <Optix/optix.h>
//#include <Optix/optix_math.h>
//#include <Optix/optixu/optixu.h>
//#include <Optix/optixu/optixu_math_namespace.h>
//#include <Optix/optixu/optixu_math.h>
//#include <Optix/optixu/optixu_math_stream_namespace.h>
//#include <Optix/optixu/optixu_math_stream.h>
//#include <Optix/optixu/optixu_vector_functions.h>
//#include <Optix/optixu/optixu_vector_types.h>
//#include <Optix/optixu/optixpp_namespace.h>
//#include <Optix/optixu/optixpp.h>

// Macros
//#include macros.h
#include "typedefs.h"

// Internal headers
#include "Structs.h"
#include "Surface.h"
#include "Buffer.h"
#include "Camera.h"
#include "Object3D.h"
#include "Scene.h"
#include "RayKernels.cuh"
#include "Renderer.h"
#include "OptixRenderer.h"
#include "Game.h"
#include "RayTracer.h"
#include "JornamEngine.h"

#endif // JORNAMENGINE