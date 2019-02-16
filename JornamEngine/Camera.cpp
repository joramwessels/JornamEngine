#include "headers.h"

namespace JornamEngine {

// Called at the start of every frame so camera can interpolate settings
void Camera::tick()
{
	// TODO
	// if (m_DoF_goal != m_DoF) m_DoF += min(DoF_increment, difference between goal and DoF);
	// if (zoom is on && m_zoom != m_zoom_goal) m_zoom += logarithmic zoom increment
}

// Sets the camera axis system given the forward and left axes
void Camera::setRotation(vec3 a_forward, vec3 a_left)
{
	if (a_forward.dot(a_left) == 0.0f && a_forward.isNonZero() && a_left.isNonZero())
	{
		m_direction = a_forward.normalized();
		m_left = a_left.normalized();
		m_up = m_direction.cross(m_left);
	}
	else if (a_forward.dot(a_left) != 0.0f)
		throw JornamEngineException("Camera",
			"The manually set camera axes are not perpendicular; the rotation hasn't been applied.\n",
			JornamEngineException::WARN);
	else if (!a_forward.isNonZero())
		throw JornamEngineException("Camera",
			"The manually set forward axis is a zero vector; the rotation hasn't been applied.\n",
			JornamEngineException::WARN);
	else if (!a_left.isNonZero())
		throw JornamEngineException("Camera",
			"The manually set left axis is a zero vector; the rotation hasn't been applied.\n",
			JornamEngineException::WARN);
}

// Sets the camera axis system assuming a horizontal left axis
void Camera::setRotation(vec3 a_forward)
{
	if (a_forward.x != 0.0f || a_forward.z != 0.0f)
	{
		if (a_forward.y == 0.0f) m_left = a_forward.cross(vec3(0.0f, 1.0f, 0.0f));
		else
		{
			vec3 projection = vec3(a_forward.x, 0.0f, a_forward.z);
			if (a_forward.y > 0.0f) m_left = a_forward.cross(projection);
			else m_left = projection.cross(a_forward);
		}
		m_direction = a_forward.normalized();
		m_left.normalize();
		m_up = m_direction.cross(m_left);
	}
	else throw JornamEngineException("Camera",
		"The manually set forward axis only has a y dimension; the rotation hasn't been applied.\n",
		JornamEngineException::WARN);
}

// Rotates the camera using the given degrees
void Camera::rotate(Quaternion q)
{
	m_direction.rotate(q.r, q.v);
	m_left.rotate(q.r, q.v);
	m_direction.normalize();
	m_left.normalize();
	m_up = m_direction.cross(m_left);
}

// Recalculates and returns the virtual screen corners
ScreenCorners Camera::getScreenCorners()
{
	ScreenCorners corners = ScreenCorners{ 0 };
	float zoomscale = m_focalPoint / m_zoom;
	vec3 screenCenter = m_location + (m_direction * m_focalPoint);
	corners.TL = screenCenter + ((m_left * m_halfScrWidth) + (m_up * m_halfScrHeight)) * zoomscale;
	corners.TR = screenCenter - ((m_left * m_halfScrWidth) - (m_up * m_halfScrHeight)) * zoomscale;
	corners.BL = screenCenter + ((m_left * m_halfScrWidth) - (m_up * m_halfScrHeight)) * zoomscale;
	return corners;
}

} // namespace Engine