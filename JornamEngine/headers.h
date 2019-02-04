#pragma once

// Disbling deprecation warnings for buffer functions in stdio
#define _CRT_SECURE_NO_DEPRECATE

// Windows
#ifdef _WIN32
#include <Windows.h>
#endif

// SDL
#include <SDL.h>

// console
#include <fcntl.h>
#include <io.h>


// C++ headers
#include <fstream>
#include <iostream>
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
#include "Surface.h"
#include "Scene.h"
#include "Renderer.h"
#include "Game.h"
#include "RayTracer.h"
#include "JornamEngine.h"