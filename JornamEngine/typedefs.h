#pragma once

// Precalculated math
#define PI		3.14159265358979323846264338327950288419716939937510582097494459072381640628620899862803482534211706798f
#define INVPI	0.31830988618379067153776752674502872406891929148091289749533468811779359526845307018022760553250617191f
#define INV2PI	0.15915494309189533576888376337251436203445964574045644874766734405889679763422653509011380276625308595f
#define INV4PI	0.07957747154594766788444188168625718101722982287022822437383367202944839881711326754505690138312654297f

namespace JornamEngine {

typedef unsigned int uint;
typedef unsigned char byte;

// Exception class for classes in the engine
class JornamException : public std::exception
{
public:
	enum LEVEL { DEBUG, INFO, WARN, ERR, FATAL };
	LEVEL m_severity;
	std::string m_class;
	std::string m_msg;
	JornamException(const std::string a_class, std::string a_msg, const LEVEL a_severity) :
		m_class(a_class), m_msg(a_msg), m_severity(a_severity) {};
	const char* what()
	{
		return ("JornamException in " + m_class + " class: " + m_msg).c_str();
	}
};

// 0x00RRGGBB (uint; 32-bit)
typedef unsigned int Color;

/*
	Multiplies a Color by a scalar and handles overflows
	@param c the original color
	@param s the value with which to scale
	@throws JornamException when a value has overflowed
*/
//Color multiplyColor(const Color c, const float s)
//{
//	uint r = (uint)(((c & 0x00FF0000) >> 16) * s);
//	uint g = (uint)(((c & 0x0000FF00) >> 8) * s);
//	uint b = (uint)((c & 0x000000FF) * s);
//	bool ro = r > 0xFF, go = g > 0xFF, bo = b > 0xFF;
//	if (ro || go || bo)
//		throw JornamException("Color", "Color value overflow (clipped)\n", JornamException::INFO);
//	if (ro) r = 0xFF;
//	if (go) g = 0xFF;
//	if (bo) b = 0xFF;
//	return (r << 16) & (g << 8) & b;
//}

enum COLOR
{
	BLACK   = 0x00000000,
	GRAY    = 0x000F0F0F,
	WHITE   = 0x00FFFFFF,
	RED     = 0x00FF0000,
	GREEN   = 0x0000FF00,
	BLUE    = 0x000000FF,
	YELLOW  = 0x000F0F00,
	CYAN    = 0x00000F0F,
	MAGENTA = 0x000F000F
};

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
	inline void operator /= (const float a) { x /= a; y /= a; z /= a; }
	inline void normalize() { float r = 1.0f / length(); x *= r; y *= r;  z *= r; }

	inline float  operator[] (const uint& i) const { return cell[i]; }
	inline float& operator[] (const uint& i)       { return cell[i]; }

	inline bool  isNonZero()          const { return (x != 0.0f || y != 0.0f || z != 0.0f); }
	inline float length()             const { return sqrt(x * x + y * y + z * z); }
	inline float sqrLength()          const { return x * x + y * y + z * z; }
	inline vec3  normalized()         const { float r = 1.0f / length(); return vec3(x * r, y * r, z * r); }
	inline float dot(const vec3& a)   const { return x * a.x + y * a.y + z*a.z; }
	inline vec3  cross(const vec3& a) const { return vec3(y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x); }
	inline const char* to_string()	  const { return (std::string("(") + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(x) + ")").c_str(); }
	inline void  print()			  const { printf("(%.2f, %.2f, %.2f)", x, y, z); }

	inline void rotate(const float& qr, const vec3& qv) { vec3 t = qv.cross(*this) * 2.0f; vec3 s = t * qr + qv.cross(t); x = s.x; y = s.y; z = s.z; }
	inline void rotate(const vec3& degrees) { rotateAroundX(degrees.x); rotateAroundY(degrees.y); rotateAroundZ(degrees.z); }
	inline void rotate(const vec3& u, const float angle)
	{
		// https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
		float cosT = cos(angle);
		float sinT = sin(angle);
		float invC = (1 - cosT);
		vec3 v = vec3(x, y, z);
		float uxInvC = u.x * invC;
		float uyInvC = u.y * invC;
		float uzInvC = u.z * invC;
		x = (u.x * uxInvC + cosT)		* v.x + (u.x * uyInvC - u.z * sinT) * v.y + (u.x * uzInvC + u.y * sinT) * v.z;
		y = (u.y * uxInvC + u.z * sinT) * v.x + (u.y * uyInvC + cosT)		* v.y + (u.y * uzInvC - u.x * sinT) * v.z;
		z = (u.z * uxInvC - u.y * sinT) * v.x + (u.z * uyInvC + u.x * sinT) * v.y + (u.z * uzInvC + cosT)		* v.z;
	}
	
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
};

inline void swap(uint* a, uint* b) {
	uint tmp = *a;
	*a = *b;
	*b = tmp;
}

} // namespace Engine