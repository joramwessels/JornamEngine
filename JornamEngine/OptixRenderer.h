#pragma once

namespace JornamEngine {

enum RAYTYPE {PRIMARY, SHADOW}; // { PRIMARY, SHADOW }

class OptixRenderer : public Renderer
{
public:
	OptixRenderer(Surface* screen, SCREENHALF renderhalf) :
		Renderer(screen, JornamEngine::USE_GPU::CUDA, renderhalf) {};
	OptixRenderer(Surface* screen) :
		Renderer(screen, JornamEngine::USE_GPU::CUDA) {};
	~OptixRenderer() { rtpContextDestroy(m_context); };
	virtual void init(Scene* scene, uint SSAA) {}; // Called once at the start of the application
	virtual void tick() {};						   // Called at the start of every frame
	virtual void render(Camera* camera) {};		   // Called at the end of every frame

protected:
	RTPcontext m_context;
	RTbuffer m_buffer;

	Scene* m_scene;			// The collection of triangles and lights to be rendered

	void initializeMaterials();
};

} // namespace Engine