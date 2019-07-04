#pragma once

namespace JornamEngine {

/*
	A position and a color (16 bytes)
	
	@param position	The location of the light
	@param color	(optional) The color of the light (defaults to white)
*/
struct Light
{
	vec3 pos;
	Color color;

	Light(vec3 position) : pos(position), color(COLOR::WHITE) {}
	Light(vec3 position, Color color) : pos(position), color(color) {}
};

/*
	An image surrounding the scene (4/8 bytes; pointer)

	@param filename	The path to the skybox file
*/
struct Skybox
{
	Surface* image;
	Skybox() : image(0) {}
	Skybox(const char* filename) : image(new Surface(filename)) {}
	~Skybox() { }//delete image;}

	/*
		Returns the skybox for given ray direction

		@param direction	The direction of the ray
		@returns			The color of the skybox at that location
	*/
	Color intersect(vec3 direction) const
	{
		if (!image) return COLOR::BLACK;
		float u = 0.5f + (atan2f(-direction.z, -direction.x) * INV2PI);
		float v = 0.5f - (asinf(-direction.y) * INVPI);
		uint x = (uint)((image->GetWidth() - 1) * u);
		uint y = (uint)((image->GetHeight() - 1) * v);
		return image->GetPixel(x, y);
	}
};

/*
	Represents a 3D environment including triangles, lights, and a skybox (8 + 3*(4/8) bytes; 3 pointers)

	@param a_context	The prime context
	@param filename		The path to the scene file
	@param camera		(optional) A pointer to the camera object
*/
class Scene
{
public:
	Scene(optix::prime::Context a_context, const char* filename, Camera* camera = 0, USE_GPU onDevice = USE_GPU::CUDA)
		: m_context(a_context),
		m_meshMap(MeshMap(m_context, &m_meshes, &m_optixModels)),
		m_textureMap(TextureMap(&m_textures)),
		m_model(m_context->createModel())
	{
		m_buffertype = (onDevice == USE_GPU::CUDA ? RTP_BUFFER_TYPE_CUDA_LINEAR : RTP_BUFFER_TYPE_HOST);
		loadScene(filename, camera);
	};
	~Scene() {}

	void addLight(Light light) { m_lights.push_back(light); }
	void readMesh(const char* filename, Transform transform, Color color);
	void loadScene(const char* filename, Camera* camera = 0);

	inline void setSkybox(Skybox skybox) { m_skybox = skybox; }
	inline Color intersectSkybox(vec3 direction) const { return m_skybox.intersect(direction); }
	vec3 interpolateNormal(uint obIdx, uint trIdx, float u, float v) const;
	Color interpolateTexture(uint obIdx, uint trIdx, float u, float v) const;

	inline const optix::prime::Context	getContext() const { return m_context; }
	inline const optix::prime::Model	getSceneModel() const { return m_model; }
	inline const Object3D				getObject(int index) const { return m_objects[index]; }
	inline const Color					getAmbientLight() const { return m_ambientLight; }

	inline const Light*					getHostLights() const { return m_lights.data(); }
	inline const JECUDA::Light*			getDeviceLights() const { return c_lights; }
	inline uint							getLightCount() const { return (uint)m_lights.size(); }

	inline const Mesh*					getHostMeshes() const { return m_meshes.data(); }
	inline const JECUDA::Mesh*			getDeviceMeshes() const { return c_meshes; }
	inline uint							getMeshCount() const { return (uint)m_meshes.size(); }

	inline const Texture*				getHostTextures() const { return m_textures.data(); }
	inline const JECUDA::Texture*		getDeviceTextures() const { return c_textures; }
	inline uint							getTextureCount() const { return (uint)m_textures.size(); }

	inline const Object3D*				getHostObjects() const { return m_objects.data(); }
	inline const JECUDA::Object3D*		getDeviceObjects() const { return c_objects; }
	inline uint							getObjectCount() const { return (uint)m_objects.size(); }

private:
	RTPbuffertype m_buffertype;
	optix::prime::Context m_context;
	optix::prime::Model m_model;

	std::vector<optix::prime::Model> m_optixModels;	// starts at idx 1
	std::vector<RTPmodel> m_rtpModels;
	std::vector<TransformMatrix> m_transforms;

	std::vector<Light> m_lights;
	std::vector<Mesh> m_meshes;		 // starts at idx 1
	std::vector<Texture> m_textures; // starts at idx 1
	std::vector<Object3D> m_objects;

	JECUDA::Light* c_lights;
	JECUDA::Mesh* c_meshes;		 // starts at idx 1
	JECUDA::Texture* c_textures; // starts at idx 1
	JECUDA::Object3D* c_objects;

	Skybox m_skybox;

	MeshMap m_meshMap;
	TextureMap m_textureMap;

	Color m_ambientLight = 0x222222;
};

/*
	Parses .scene files to load the Scene object

	@param scene	A pointer to the Scene object to initialize
*/
class SceneParser
{
public:
	SceneParser(Scene* scene) : m_scene(scene) {}
	void parseScene(const char* filename, Camera* camera = 0);

	uint2 parseDimensions(const char* line);
	void parseObject(const char* line);
	void parseLight(const char* line);
	void parseSkybox(const char* line);
	void parseCamera(const char* line, Camera* camera);

	vec3 parseVec3(const char* line, uint col);
	Color parseColor(const char* line, uint col);
	float parseFloat(const char* line, uint col);

	uint skipWhiteSpace(const char* line, uint i = 0);
	uint skipExpression(const char* line, uint i = 0);
private:
	Scene* m_scene;
};

} // namespace Engine