#pragma once

// Precalculated math
#define PI		3.14159265358979323846264338327950288419716939937510582097494459072381640628620899862803482534211706798f
#define INVPI	0.31830988618379067153776752674502872406891929148091289749533468811779359526845307018022760553250617191f
#define INV2PI	0.15915494309189533576888376337251436203445964574045644874766734405889679763422653509011380276625308595f
#define INV4PI	0.07957747154594766788444188168625718101722982287022822437383367202944839881711326754505690138312654297f

typedef unsigned int uint;
typedef unsigned char byte;

// 0x00RRGGBB (uint; 32-bit)
typedef unsigned int Color;

// Vector of 3 floats (12 bytes)
struct vec3
{
	union { struct { float x, y, z; }; float cell[3]; };

	vec3()                          : x(0), y(0), z(0) {};
	vec3(float s)                   : x(s), y(s), z(s) {};
	vec3(float x, float y, float z) : x(x), y(y), z(z) {};

	inline vec3 operator - ()               const { return vec3(-x, -y, -z); }
	inline vec3 operator + (const vec3& a)  const { return vec3(x + a.x, y + a.y, z + a.z); }
	inline vec3 operator - (const vec3& a)  const { return vec3(x - a.x, y - a.y, z - a.z); }
	inline vec3 operator * (const vec3& a)  const { return vec3(x * a.x, y * a.y, z * a.z); }
	inline vec3 operator * (const float& a) const { return vec3(x * a, y * a, z * a); }
	inline vec3 operator / (const float& a) const { return vec3(x / a, y / a, z / a); }

	inline void operator -= (const vec3& a) { x -= a.x; y -= a.y; z -= a.z; }
	inline void operator += (const vec3& a) { x += a.x; y += a.y; z += a.z; }
	inline void operator *= (const vec3& a) { x *= a.x; y *= a.y; z *= a.z; }
	inline void operator *= (const float a) { x *= a;   y *= a;   z *= a; }
	inline void normalize() { float r = 1.0f / length(); x *= r; y *= r;  z *= r; }

	inline float  operator[] (const uint& i) const { return cell[i]; }
	inline float& operator[] (const uint& i)       { return cell[i]; }

	inline bool  isNonZero()          const { return (x != 0.0f || y != 0.0f || z != 0.0f); }
	inline float length()             const { return sqrt(x * x + y * y + z * z); }
	inline float sqrLength()          const { return x * x + y * y + z * z; }
	inline vec3  normalized()         const { float r = 1.0f / length(); return vec3(x * r, y * r, z * r); }
	inline float dot(const vec3& a)   const { return x * a.x + y * a.y + z*a.z; }
	inline vec3  cross(const vec3& a) const { return vec3(y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x); }

	inline void  rotateAroundX(const float& deg)
	{
		float tempy = y, tempz = z;
		y = tempy * cosf(deg * PI / (180)) - tempz * sinf(deg * PI / (180));
		z = tempy * sinf(deg * PI / (180)) + tempz * cosf(deg * PI / (180));
	}
	inline void rotateAroundY(const float& deg)
	{
		float tempx = x, tempz = z;
		x = tempx * cosf(deg * PI / (180)) + tempz * sinf(deg * PI / (180));
		z = -(tempx * sinf(deg * PI / (180))) + tempz * cos(deg * PI / (180));
	}
	inline void rotateAroundZ(const float& deg)
	{
		float tempx = x, tempy = y;
		x = tempx * cos(deg * PI / 180) - tempy * sin(deg * PI / 180);
		y = tempx * sin(deg * PI / 180) + tempy * cos(deg * PI / 180);
	}
	inline void rotate(const vec3& degrees)
	{
		rotateAroundX(degrees.x);
		rotateAroundY(degrees.y);
		rotateAroundZ(degrees.z);
	}
};