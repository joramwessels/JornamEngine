#pragma once

namespace JornamEngine {

class Camera
{
	vec3 m_location;
	vec3 m_rotation;
public:
	Camera();
	~Camera();

	void setLocation(vec3 location) { m_location = location; }
	void setRotation(vec3 rotation) { m_rotation = rotation; }
	void move(vec3 direction) { m_location += direction; }
	void rotate(vec3 degrees);

	void getTopLeft();
	void getTopright();
	void getBottomLeft();
	void getXAxis();
	void getYAxis();
	void getZAxis();
};

} // namespace Engine