#pragma once

namespace JornamEngine {

union ScreenCorners { struct { vec3 TL; vec3 TR; vec3 BL; }; float cell[9]; };

class Camera
{
protected:
	float m_focalPoint = 1.0f;
	float m_zoom = 1.0f;

	// Camera axes
	vec3 m_location;
	vec3 m_direction;
	vec3 m_left;
	vec3 m_up;

	// Virtual screen
	float m_halfScrWidth;
	float m_halfScrHeight;

public:
	Camera(const float scrWidth, const float scrHeight) :
		m_halfScrWidth(scrWidth), m_halfScrHeight(scrHeight), m_location(vec3(0.0f)),
		m_direction(vec3(0.0f, 0.0f, -1.0f)), m_left(vec3(-1.0f, 0.0f, 0.0f)),
		m_up(vec3(0.0f, -1.0f, 0.0f)), m_focalPoint(1.0f), m_zoom(1.0f) {};
	~Camera() {};

	void setLocation(vec3 location) { m_location = location; }
	void setRotation(vec3 forward, vec3 left);
	void setRotation(vec3 forward);
	void setFocalPoint(float focalPoint) { m_focalPoint = focalPoint; }
	void setZoom(float zoom) { m_zoom = zoom; }

	void tick();
	void move(vec3 direction, float speed) { m_location += direction * speed; }
	void moveForward(float speed) { m_location += m_direction * speed; }
	void moveLeft(float speed) { m_location += m_left * speed; }
	void moveUp(float speed) { m_location += m_up * speed; }
	void rotate(vec3 axis, float angle);

	ScreenCorners getScreenCorners() const;
	vec3 getLocation() const { return m_location; }
	vec3 getForward() const { return m_direction; }
	vec3 getLeft() const { return m_left; }
	vec3 getUp() const { return m_up; }
};

} // namespace Engine