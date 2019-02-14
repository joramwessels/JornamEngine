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
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <curand.h>
//#include <curand_kernel.h>

// Macros
//#include macros.h
#include "typedefs.h"

// Internal headers
#include "Structs.h"
#include "Surface.h"
#include "Scene.h"
#include "Camera.h"
#include "Renderer.h"
#include "Game.h"
#include "RayTracer.h"
#include "JornamEngine.h"

#endif // JORNAMENGINE