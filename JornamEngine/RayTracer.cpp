#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(Scene* a_scene, uint a_SSAA)
{
	m_scene = a_scene;
	m_SSAA = a_SSAA;

	rtContextCreate(&m_context);
	rtContextSetEntryPointCount(m_context, 2);

	initializeMaterials();

	// Initializing rt output buffer
	rtBufferCreate(m_context, RT_BUFFER_OUTPUT, &m_buffer);
	rtBufferSetFormat(m_buffer, RT_FORMAT_USER);
	rtBufferSetElementSize(m_buffer, sizeof(Color));
	rtBufferSetSize2D(m_buffer, m_scrwidth, m_scrheight);

	RTprogram pinholeCamera = 0;
	RTprogram thinLensCamera = 0;
	RTprogram exceptionProgram = 0;

	rtContextSetRayGenerationProgram(m_context, 0, pinholeCamera);
	rtContextSetRayGenerationProgram(m_context, 1, thinLensCamera);
	rtContextSetExceptionProgram(m_context, 0, exceptionProgram);
	rtContextSetExceptionProgram(m_context, 1, exceptionProgram);
}

// Called at the start of every frame
void RayTracer::tick()
{

}

void RayTracer::render(Camera* camera)
{
	rtContextLaunch2D(m_context, 0, m_scrwidth, m_scrheight);

	// Copying device buffer to host Surface buffer
	void* surfaceBuffer;
	rtBufferMap(m_buffer, &surfaceBuffer);
	Color* buffer = (Color*)surfaceBuffer;
	memcpy(m_screen->GetBuffer(), buffer, m_scrwidth * m_scrheight * sizeof(Color)); // TODO does m_screen.Plot() work after rtBufferUnmap?
	rtBufferUnmap(m_buffer);
}

} // namespace Engine