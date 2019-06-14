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
__device__ struct CudaLight { float3 pos; unsigned int color; };

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
	Scene(optix::prime::Context a_context, const char* filename, Camera* camera = 0)
		: m_context(a_context),
		m_meshMap(MeshMap(m_context, &m_meshes, c_meshes)),
		m_model(m_context->createModel())
		{ loadScene(filename, camera); };
	~Scene() {}

	void addLight(Light light) { m_lights.push_back(light); }
	void readObject(const char* filename, Transform transform);
	void addObject(std::vector<float> vertices, std::vector<uint> indices, std::vector<vec3> normals, Transform transform, Color color);
	void loadScene(const char* filename, Camera* camera = 0);

	inline void setSkybox(Skybox skybox) { m_skybox = skybox; }
	inline Color intersectSkybox(vec3 direction) const { return m_skybox.intersect(direction); }

	inline const optix::prime::Context	getContext() const { return m_context; }
	inline const optix::prime::Model	getSceneModel() const { return m_model; }
	inline const Object3D				getObject(int index) const { return m_objects[index]; }
	inline const Color					getAmbientLight() const { return m_ambientLight; }

	inline const Light*					getHostLights() const { return m_lights.data(); }
	inline const CudaLight*				getDeviceLights() const { return c_lights; }
	inline const uint					getLightCount() const { return m_lights.size(); }

	inline const Mesh*					getHostMeshes() const { return m_meshes.data(); }
	inline const CudaMesh*				getDeviceMeshes() const { return c_meshes; }
	inline const uint					getMeshCount() const { return m_meshes.size(); }

	//inline const Texture*				getHostTextures() const { return m_textures.data(); } // TODO
	//inline const CudaTexture*			getDeviceTextures() const { return c_textures; }	  // TODO
	//inline const uint					getTextureCount() const { return m_textures.size(); } // TODO

	inline const Object3D*				getHostObjects() const { return m_objects.data(); }
	inline const CudaObject3D*			getDeviceObjects() const { return c_objects; }
	inline const uint					getObjectCount() const { return m_objects.size(); }

private:
	RTPbuffertype m_buffertype = RTP_BUFFER_TYPE_HOST;
	optix::prime::Context m_context;
	optix::prime::Model m_model;

	std::vector<RTPmodel> m_optixModels;
	std::vector<TransformMatrix> m_transforms;

	std::vector<Light> m_lights;
	std::vector<Mesh> m_meshes;
	//std::vector<Texture> m_textures; // TODO
	std::vector<Object3D> m_objects;

	CudaLight* c_lights;
	CudaMesh* c_meshes;
	//CudaTexture* c_textures; // TODO
	CudaObject3D* c_objects;

	Skybox m_skybox;

	MeshMap m_meshMap;

	Color m_ambientLight = 0x020205;
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
	void parseTriangle(const char* line);
	void parsePlane(const char* line);
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