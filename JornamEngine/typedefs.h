#pragma once

// Precalculated math
#define PI		3.14159265358979323846264338327950288419716939937510582097494459072381640628620899862803482534211706798f
#define INVPI	0.31830988618379067153776752674502872406891929148091289749533468811779359526845307018022760553250617191f
#define INV2PI	0.15915494309189533576888376337251436203445964574045644874766734405889679763422653509011380276625308595f
#define INV4PI	0.07957747154594766788444188168625718101722982287022822437383367202944839881711326754505690138312654297f

// Exceptions and logging
#define JE_DEBUG_LVL 2
#define JE_LOG_LVL 1

namespace JornamEngine {

typedef unsigned char byte;
typedef unsigned short dbyte;
typedef unsigned int uint;

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
		if (m_severity == DEBUG) return ("DEBUG: " + m_class + " class: " + m_msg).c_str();
		if (m_severity == INFO) return ("INFO: " + m_class + " class: " + m_msg).c_str();
		if (m_severity == WARN) return ("WARNING in " + m_class + " class: " + m_msg).c_str();
		if (m_severity == ERR) return ("JornamException in " + m_class + " class: " + m_msg).c_str();
		if (m_severity == FATAL) return ("FATAL JornamException in " + m_class + " class: " + m_msg).c_str();
		else return ("UNKNOWN_SEVERITY (" + std::to_string((int)m_severity) + ") in " + m_class + " class: " + m_msg).c_str();
	}
};

inline void logDebug(const char* a_class, const char* a_msg, const JornamException::LEVEL a_severity)
{
	JornamException e = JornamException(a_class, a_msg, a_severity);
	if (a_severity >= JE_LOG_LVL) printf("%s", e.what());
	if (a_severity >= JE_DEBUG_LVL) { throw e; }
}

// 0x00RRGGBB (4 bytes)
struct Color
{
	union { uint hex; struct { byte b, g, r, x; }; };

	inline void operator+=(const Color& c)
	{
		uint ar = r + c.r, ag = g + c.g, ab = b + c.b;
		checkOverflow(ar, ag, ab);
		r = (byte)ar; g = (byte)ag; b = (byte)ab;
	}

	inline Color operator*(const float& s) const
	{
		uint ar = (uint)((float)((hex >> 16) & 0xFF) * s);
		uint ag = (uint)((float)((hex >> 8) & 0xFF) * s);
		uint ab = (uint)((float)(hex & 0xFF) * s);
		checkOverflow(ar, ag, ab);
		return ((ar << 16) & 0xFF0000) | ((ag << 8) & 0xFF00) | (ab & 0xFF);
	}

	inline Color operator*(const Color& c) const
	{
		uint ar = ((hex >> 16) & 0xFF) * ((c.hex >> 16) & 0xFF);
		uint ag = ((hex >> 8) & 0xFF) * ((c.hex >> 8) & 0xFF);
		uint ab = (hex & 0xFF) * (c.hex & 0xFF);
		checkOverflow(ar, ag, ab);
		return ((ar << 16) & 0xFF0000) | ((ag << 8) & 0xFF00) | (ab & 0xFF);
	}

	inline Color directIllumination(const Color& c, const float& s) const
	{
		uint ar = (uint)((float)(((hex >> 16) & 0xFF) * ((c.hex >> 16) & 0xFF)) * s);
		uint ag = (uint)((float)(((hex >> 8) & 0xFF) * ((c.hex >> 8) & 0xFF)) * s);
		uint ab = (uint)((float)((hex & 0xFF) * (c.hex & 0xFF)) * s);
		checkOverflow(ar, ag, ab);
		return ((ar << 16) & 0xFF0000) | ((ag << 8) & 0xFF00) | (ab & 0xFF);
	}

	inline const void checkOverflow(uint &ar, uint &ag, uint &ab) const
	{
		bool ro = ar > 0xFF, go = ag > 0xFF, bo = ab > 0xFF;
		if (ro) { ar = 0xFF;
			logDebug("Color", "Color value overflow (r clipped)\n", JornamException::DEBUG); }
		if (go) { ag = 0xFF;
			logDebug("Color", "Color value overflow (g clipped)\n", JornamException::DEBUG); }
		if (bo) { ab = 0xFF;
			logDebug("Color", "Color value overflow (b clipped)\n", JornamException::DEBUG); }
	}

	Color() {}
	Color(uint hex) : hex(hex) {}
	Color(byte r, byte g, byte b) : r(r), g(g), b(b) {}
};

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

// A row-dominant 4x3 affine transformation matrix (48 bytes)
struct TransformMatrix
{
	float t0, t1, t2, t3;
	float t4, t5, t6, t7;
	float t8, t9, t10, t11;

	// Default constructor creates the identity matrix
	TransformMatrix() :
		t0(1.0f), t1(0.0f), t2(0.0f), t3(0.0f),
		t4(0.0f), t5(1.0f), t6(0.0f), t7(0.0f),
		t8(0.0f), t9(0.0f), t10(1.0f), t11(0.0f) {}

	// Scaling matrix
	TransformMatrix(vec3 scale) : TransformMatrix() { t0 = scale.x; t5 = scale.y; t10 = scale.z; }

	// Rotation matrix (axis should be normalized)
	TransformMatrix(vec3 axis, float angle)
		: TransformMatrix() {
		rotate(axis, angle);
	}

	// Translation matrix (when axis or angle are 0), otherwise axis should be normalized
	TransformMatrix(vec3 axis, float angle, vec3 pos)
		: TransformMatrix(axis, angle) {
		translate(pos);
	}

	// Full Transformation matrix (axis should be normalized)
	TransformMatrix(vec3 a_axis, float a_angle, vec3 a_pos, vec3 a_scale)
		: TransformMatrix(a_axis, a_angle, a_pos) {
		scale(a_scale);
	}

	// Scales the matrix
	inline void scale(vec3 scale)
	{
		t0 *= scale.x; t4 *= scale.x; t8 *= scale.x;
		t1 *= scale.y; t5 *= scale.y; t9 *= scale.y;
		t2 *= scale.z; t6 *= scale.z; t10 *= scale.z;
	}

	// Rotates the matrix (axis should be normalized)
	inline void rotate(vec3 axis, float angle)
	{
		if ((axis.x == 0.0f && axis.y == 0.0f && axis.z == 0.0f) || angle == 0.0f) return;
		float cosT = cos(angle), sinT = sin(angle), mcosT = 1 - cosT;
		t0 = cosT + axis.x * axis.x * mcosT;
		t1 = axis.x * axis.y * mcosT - axis.z * sinT;
		t2 = axis.x * axis.z * mcosT + axis.y * sinT;
		t4 = axis.x * axis.y * mcosT + axis.z * sinT;
		t5 = cosT + axis.y * axis.y * mcosT;
		t6 = axis.y * axis.z * mcosT - axis.x * sinT;
		t8 = axis.x * axis.z * mcosT - axis.y * sinT;
		t9 = axis.y * axis.z * mcosT + axis.x * sinT;
		t10 = cosT + axis.z * axis.z * mcosT;
	}

	// Translates the matrix
	inline void translate(vec3 pos) { t3 += pos.x; t7 += pos.y; t11 += pos.z; }

	inline TransformMatrix operator*(TransformMatrix &a) const
	{
		TransformMatrix result = TransformMatrix();
		result.t0 = t0 * a.t0 + t1 * a.t4 + t2 * a.t8;
		result.t1 = t0 * a.t1 + t1 * a.t5 + t2 * a.t9;
		result.t2 = t0 * a.t2 + t1 * a.t6 + t2 * a.t10;
		result.t3 = t0 * a.t3 + t1 * a.t7 + t2 * a.t11 + t3;
		result.t4 = t4 * a.t0 + t5 * a.t4 + t6 * a.t8;
		result.t5 = t4 * a.t1 + t5 * a.t5 + t6 * a.t9;
		result.t6 = t4 * a.t2 + t5 * a.t6 + t6 * a.t10;
		result.t7 = t4 * a.t3 + t5 * a.t7 + t6 * a.t11 + t7;
		result.t8 = t8 * a.t0 + t9 * a.t4 + t10 * a.t8;
		result.t9 = t8 * a.t1 + t9 * a.t5 + t10 * a.t9;
		result.t10 = t8 * a.t2 + t9 * a.t6 + t10 * a.t10;
		result.t11 = t8 * a.t3 + t9 * a.t7 + t10 * a.t11 + t11;
		return result;
	}
};

struct OptixRay
{
	vec3 origin, direction;
	OptixRay() : origin(vec3(0.0f)), direction(vec3(0.0f)) {};
	OptixRay(vec3 ori, vec3 dir) : origin(ori), direction(dir) {};
};

struct OptixHit { float rayDistance; int triangleIdx; int instanceIdx; float u; float v; };

struct OptixModel
{
	optix::prime::Model handle;
	std::vector<vec3> N;
	Color color;
	uint triCount;

	OptixModel() {}
	OptixModel(optix::prime::Model model, std::vector<vec3> N, Color color) : handle(model), N(N), color(color) {}

	RTPmodel getRTPmodel() const { return handle->getRTPmodel(); }
	void setTriangles(std::vector<uint> indices, std::vector<float> vertices, RTPbuffertype type)
	{
		handle->setTriangles(
			indices.size() / 3, type, indices.data(),
			vertices.size() / 3, type, vertices.data()
		);
		handle->update(0);
	}
};

} // namespace Engine