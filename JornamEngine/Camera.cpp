/*
	file:			Camera.cpp
	last modified:	18-02-2019
	description:	Provides a camera class

	@author Joram Wessels
	@version 0.1
*/
#include "headers.h"

namespace JornamEngine {

// Called at the start of every frame so camera can interpolate settings
void Camera::tick()
{
	// TODO DoF temporal interpolation
	// if (m_DoF_goal != m_DoF) m_DoF += min(DoF_increment, difference between goal and DoF);
	// if (zoom is on && m_zoom != m_zoom_goal) m_zoom += logarithmic zoom increment
}

/*
	Sets the camera axis system given the forward and left axes

	@param forward	A vector pointing directly forwards
	@param left		A vector pointing to the left
*/
void Camera::setRotation(vec3 a_forward, vec3 a_left)
{
	if (a_forward.dot(a_left) == 0.0f && a_forward.isNonZero() && a_left.isNonZero())
	{
		m_direction = a_forward.normalized();
		m_left = a_left.normalized();
		m_up = m_direction.cross(m_left);
	}
	else if (a_forward.dot(a_left) != 0.0f)
		logger.logDebug("Camera",
			"The manually set camera axes are not perpendicular; the rotation hasn't been applied.",
			JornamException::WARN);
	else if (!a_forward.isNonZero())
		logger.logDebug("Camera",
			"The manually set forward axis is a zero vector; the rotation hasn't been applied.",
			JornamException::WARN);
	else if (!a_left.isNonZero())
		logger.logDebug("Camera",
			"The manually set left axis is a zero vector; the rotation hasn't been applied.",
			JornamException::WARN);
}

/*
	Sets the camera axis system assuming a horizontal left axis

	@param forward	A vector poiting directly forward
*/
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
	else logger.logDebug("Camera",
		"The manually set forward axis only has a y dimension; the rotation hasn't been applied.",
		JornamException::WARN);
}

/*
	Rotates the camera using the given axis and degrees

	@param axis		The axis around which to rotate
	@param angle	The angle of rotation
*/
void Camera::rotate(vec3 axis, float angle)
{
	angle *= m_rotationSensitivity;
	if (m_invertAxes) angle = -angle;
	m_direction.rotate(axis, angle);
	m_left.rotate(axis, angle);
	m_direction.normalize();
	m_left.normalize();
	m_up = m_direction.cross(m_left);
}

/*
	Rotates the camera around the X-axis (pitch)

	@param angle	The angle of rotation
*/
void Camera::rotateX(float angle)
{
	angle *= m_rotationSensitivity;
	if (m_invertAxes) angle = -angle;
	m_direction.rotate(m_left, angle);
	m_direction.normalize();
	m_up = m_direction.cross(m_left);
}

/*
	Rotates the camera around the Y-axis (yaw)

	@param angle	The angle of rotation
*/
void Camera::rotateY(float angle)
{
	angle *= m_rotationSensitivity;
	if (m_invertAxes) angle = -angle;
	if (m_lockRoll)
	{
		m_direction.rotate(JE_AXIS_DOWN, angle);
		m_direction.normalize();
		m_left.rotate(JE_AXIS_DOWN, angle);
		m_left.normalize();
	}
	else
	{
		m_direction.rotate(m_up, angle);
		m_direction.normalize();
		m_left.rotate(m_up, angle);
		m_left.normalize();
		m_up = m_direction.cross(m_left);
	}
}

/*
	Recalculates and returns the virtual screen corners

	@returns	The virtual screen coordinates as a ScreenCorners struct
*/
ScreenCorners Camera::getScreenCorners() const
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