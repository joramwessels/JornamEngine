#include "headers.h"

namespace JornamEngine {

// Called at the start of every frame so camera can interpolate settings
void Camera::tick()
{
	// TODO
	// if (m_DoF_goal != m_DoF) m_DoF += min(DoF_increment, difference between goal and DoF);
	// if (zoom is on && m_zoom != m_zoom_goal) m_zoom += logarithmic zoom increment
	calculateTangents();
}

// Rotates the camera using the given degrees
void Camera::rotate(vec3 degrees)
{
	// TODO
}

// Recalculates the two tangent vectors (left and up)
void Camera::calculateTangents()
{
	m_direction.normalize();
	m_left.x = m_direction.x;
	m_left.y = 0;
	m_left.z = m_direction.z;
	m_left.rotateAroundY(90.0f);
	m_left.normalize();
	m_up = m_direction.cross(m_left);
}

// Recalculates and returns the virtual screen corners
Corners Camera::getScreenCorners()
{
	calculateTangents();
	Corners corners = Corners{ 0 };
	float zoomscale = m_focalPoint / m_zoom;
	vec3 screenCenter = m_location + (m_direction * m_focalPoint);
	corners.TL = screenCenter - (m_left * zoomscale * m_scrWidth) - (m_up * zoomscale * m_scrHeight);
	corners.TR = screenCenter + (m_left * zoomscale * m_scrWidth) - (m_up * zoomscale * m_scrHeight);
	corners.BL = screenCenter - (m_left * zoomscale * m_scrWidth) + (m_up * zoomscale * m_scrHeight);
	return corners;
}

} // namespace Engine