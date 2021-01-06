// Based on https://github.com/matt77hias/cu-smallpt.git and https://github.com/seifeddinedridi/smallvpt.git
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "curand_kernel.h"

#include <stdlib.h> 
#include <stdio.h>

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance() 

inline void HandleError(cudaError_t err) {
    if (cudaSuccess != err) {
        printf("%s in %s at line %d\n",
            cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

struct Vec {
    float x, y, z;                  // position, also color (r,g,b) 
    __host__ __device__ explicit Vec() { x = 0.f; y = 0.f; z = 0.f; }
    __host__ __device__ explicit Vec(float v) { x = v; y = v; z = v; }
    __host__ __device__ explicit Vec(float x_ = 0.f, float y_ = 0.f, float z_ = 0.f) { x = x_; y = y_; z = z_; }
    Vec(const Vec& vec) noexcept = default;
    Vec(Vec && vec) noexcept = default;
    ~Vec() = default;
    __device__ Vec& operator=(const Vec& b) { this->x = b.x; this->y = b.y; this->z = b.z; return *this; }
    __device__ const Vec operator+(const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }
    __device__ const Vec operator-(const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }
    __device__ const Vec operator*(float b) const { return Vec(x * b, y * b, z * b); }
    __device__ const Vec mult(const Vec& b) const { return Vec(x * b.x, y * b.y, z * b.z); }
    __host__ __device__ float len() const { return sqrt(x * x + y * y + z * z); }
    __host__ __device__ Vec& norm() { float inv_len = 1.f / len(); this->x *= inv_len; this->y *= inv_len; this->z *= inv_len; return *this; }
    __device__ float dot(const Vec& b) const { return x * b.x + y * b.y + z * b.z; } // cross: 
    __device__ Vec operator%(Vec& b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
    __device__ Vec operator%(const Vec& b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

struct Ray { 
    Vec o, d; 
    __host__ __device__ explicit Ray() : o(Vec(0.f, 0.f, 0.f)), d(Vec(0.f, 0.f, 0.f)) {}
    __host__ __device__ explicit Ray(Vec o_, Vec d_) noexcept  : o(o_), d(d_) {}
    Ray(const Ray& ray) noexcept = default;
    Ray(Ray && ray) noexcept = default;
    ~Ray() = default;
    __device__ Ray& operator=(const Ray& r) { this->o = r.o; this->d = r.d; return *this; }
};

struct Sphere {
    float rad;        // radius
    Vec p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
    __host__ __device__ explicit Sphere(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
    __device__ float intersect(const Ray& r, float* tin = NULL, float* tout = NULL) const { // returns distance, 0 if nohit
        Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        float t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
        if (det < 0.f) return 0; else det = sqrt(det);
        if (tin && tout) { *tin = (b - det <= 0.f) ? 0.f : b - det; *tout = b + det; }
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0.f);
    }
};

Sphere spheres[] = {//Scene: radius, position, emission, color, material 
    Sphere(26.5f, Vec(27.f, 18.5f, 78.f), Vec(0.f, 0.f, 0.f), Vec(1.f, 1.f, 1.f) * .75f, SPEC),//Mirr
    Sphere(12.f, Vec(70.f, 43.f, 78.f), Vec(0.f, 0.f, 0.f), Vec(0.27f,0.8f,0.8f), REFR),//Glas
    Sphere(8.f, Vec(55.f, 87.f, 78.f), Vec(0.f, 0.f, 0.f),  Vec(1,1,1) * .75f, DIFF), //Lite
    Sphere(4.f, Vec(55.f, 80.f, 78.f), Vec(10.f,10.f,10.f),  Vec(0.f, 0.f, 0.f), DIFF) //Lite
};

__device__ __host__ inline float clamp(float x) { return x < 0.f ? 0.f : x>1.f ? 1.f : x; }
inline int toInt(float x) { return int(pow(clamp(x), 1.f / 2.2f) * 255.f + .5f); }
__device__ inline bool intersect(const Sphere* spheres, size_t n_sphere, const Ray& r, float& t, int& id, float tmax = 1e20) {
    float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = tmax;
    for (int i = int(n); i--;) if ((d = spheres[i].intersect(r)) && d < t) { t = d; id = i; }
    return t < inf;
}
__device__ inline float sampleSegment(float epsilon, float sigma, float smax) {
    return -log(1.f - epsilon * (1.f - exp(-sigma * smax))) / sigma;
}
__device__ inline Vec sampleSphere(float e1, float e2) {
    float z = 1.f - 2.f * e1, sint = sqrt(1.f - z * z);
    return Vec(cos(2.f * CUDART_PI_F * e2) * sint, sin(2.f * CUDART_PI_F * e2) * sint, z);
}
__device__ inline Vec sampleHG(float g, float e1, float e2) {
    //float s=2.0f*e1-1.0f, f = (1.0f-g*g)/(1.0f+g*s), cost = 0.5f*(1.0f/g)*(1.0f+g*g-f*f), sint = sqrt(1.0f-cost*cost);
    float s = 1.f - 2.f * e1, cost = (s + 2.f * g * g * g * (-1.0 + e1) * e1 + g * g * s + 2.f * g * (1.f - e1 + e1 * e1)) / ((1.f + g * s) * (1.f + g * s)), sint = sqrt(1.f - cost * cost);
    return Vec(cos(2.f * CUDART_PI_F * e2) * sint, sin(2.f * CUDART_PI_F * e2) * sint, cost);
}
__device__ inline void generateOrthoBasis(Vec& u, Vec& v, Vec w) {
    Vec coVec = w;
    if (fabs(w.x) <= fabs(w.y))
        if (fabs(w.x) <= fabs(w.z)) coVec = Vec(0.f, -w.z, w.y);
        else coVec = Vec(-w.y, w.x, 0.f);
    else if (fabs(w.y) <= fabs(w.z)) coVec = Vec(-w.z, 0.f, w.x);
    else coVec = Vec(-w.y, w.x, 0.f);
    coVec.norm();
    u = w % coVec, v = w % u;
}

__device__ inline float scatter(const Ray& r, Ray* sRay, float tin, float tout, float& s, const float& sigma_s, curandState_t* rand_state) {
    s = sampleSegment(curand_uniform(rand_state), sigma_s, tout - tin);
    Vec x = r.o + r.d * tin + r.d * s;
    //Vec dir = sampleSphere(curand_uniform(rand_state), curand_uniform(rand_state)); //Sample a direction ~ uniform phase function
    Vec dir = sampleHG(-0.5f, curand_uniform(rand_state), curand_uniform(rand_state)); //Sample a direction ~ Henyey-Greenstein's phase function
    Vec u(0.f, 0.f, 0.f), v(0.f, 0.f, 0.f);
    generateOrthoBasis(u, v, r.d);
    dir = u * dir.x + v * dir.y + r.d * dir.z;
    if (sRay)	*sRay = Ray(x, dir);
    return (1.0f - exp(-sigma_s * (tout - tin)));
}

__device__ Vec radiance(const Sphere* spheres, size_t n_sphere, const Ray& r, int depth, curandState_t* rand_state) {
    float t;                               // distance to intersection
    int id = 0;                               // id of intersected object
    float tnear, tfar, scaleBy = 1.0, absorption = 1.0;
    const Sphere homoMedium(300.f, Vec(50.f, 50.f, 80.f), Vec(0.f, 0.f, 0.f), Vec(0.f, 0.f, 0.f), DIFF);
    const float sigma_s = 0.009f, sigma_a = 0.006f, sigma_t = sigma_s + sigma_a;
    bool intrsctmd = homoMedium.intersect(r, &tnear, &tfar) > 0;
    if (intrsctmd) {
        Ray sRay;
        float s, ms = scatter(r, &sRay, tnear, tfar, s, sigma_s, rand_state), prob_s = ms;
        scaleBy = 1.0 / (1.0 - prob_s);
        if (curand_uniform(rand_state) <= prob_s) {// Sample surface or volume?
            if (!intersect(spheres, n_sphere, r, t, id, tnear + s))
                return radiance(spheres, n_sphere, sRay, ++depth, rand_state) * ms * (1.0 / prob_s);
            scaleBy = 1.0;
        }
        else
            if (!intersect(spheres, n_sphere, r, t, id)) return Vec(0.f, 0.f, 0.f);
        if (t >= tnear) {
            float dist = (t > tfar ? tfar - tnear : t - tnear);
            absorption = exp(-sigma_t * dist);
        }
    }
    else
        if (!intersect(spheres, n_sphere, r, t, id)) return Vec(0.f, 0.f, 0.f);
    const Sphere& obj = spheres[id];        // the hit object
    Vec x = r.o + r.d * t, n = Vec(x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c, Le = obj.e;
    float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
    if (++depth > 5) if (curand_uniform(rand_state) < p) { f = f * (1 / p); }
    else return Vec(0.f, 0.f, 0.f); //R.R.
    if (n.dot(nl) > 0 || obj.refl != REFR) { f = f * absorption; Le = obj.e * absorption; }// no absorption inside glass
    else scaleBy = 1.0;
    if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
        float r1 = 2 * CUDART_PI_F * curand_uniform(rand_state), r2 = curand_uniform(rand_state), r2s = sqrt(r2);
        Vec w = nl, u = Vec((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1.f, 1.f, 1.f)) % w).norm(), v = w % u;
        Vec d = Vec(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
        return (Le + f.mult(radiance(spheres, n_sphere, Ray(x, d), depth, rand_state))) * scaleBy;
    }
    else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
        return (Le + f.mult(radiance(spheres, n_sphere, Ray(x, r.d - n * 2 * n.dot(r.d)), depth, rand_state))) * scaleBy;
    Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
    bool into = n.dot(nl) > 0;                // Ray from outside going in?
    float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
    if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0)    // Total internal reflection
        return (Le + f.mult(radiance(spheres, n_sphere, reflRay, depth, rand_state)));
    Vec tdir = Vec(r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
    float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
    float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
    return (Le + (depth > 2 ? (curand_uniform(rand_state) < P ?   // Russian roulette
        radiance(spheres, n_sphere, reflRay, depth, rand_state) * RP : f.mult(radiance(spheres, n_sphere, Ray(x, tdir), depth, rand_state) * TP)) :
        radiance(spheres, n_sphere, reflRay, depth, rand_state) * Re + f.mult(radiance(spheres, n_sphere, Ray(x, tdir), depth, rand_state) * Tr))) * scaleBy;
}

__global__ static void render_kernel(const Sphere* spheres, size_t n_sphere, const Ray& cam, size_t w, size_t h, Vec* Ls, int spp) {
    const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t offset = x + y * blockDim.x * gridDim.x;
    const float inv_spp = 1.0f / float(spp);
    if (x >= w || y >= h) return;

    curandState rand_state;
    curand_init(offset, 0u, 0u, &rand_state);
    
    const double fov = 0.5135f;
    const Vec cx = Vec(w * fov / h, 0.0f, 0.0f);
    const Vec cy = Vec(Vec(w * fov / h, 0.0f, 0.0f) % cam.d).norm() * fov;
    
    for (size_t sy = 0u, i = (h - 1u - y) * w + x; sy < 2u; ++sy) { // 2 subpixel row
        for (size_t sx = 0u; sx < 2u; ++sx) { // 2 subpixel column
            Vec L(0.f,0.f,0.f);
            for (size_t s = 0u; s < spp; ++s) { // samples per subpixel
                const double u1 = 2.0 * curand_uniform_double(&rand_state);
                const double u2 = 2.0 * curand_uniform_double(&rand_state);
                const double dx = (u1 < 1.0) ? sqrt(u1) - 1.0 : 1.0 - sqrt(2.0 - u1);
                const double dy = (u2 < 1.0) ? sqrt(u2) - 1.0 : 1.0 - sqrt(2.0 - u2);
                const Vec d = cx * (((sx + 0.5 + dx) * 0.5 + x) / w - 0.5) +
                    cy * (((sy + 0.5 + dy) * 0.5 + y) / h - 0.5) + cam.d;

                L = L + radiance(spheres, n_sphere, cam, 0, &rand_state) * inv_spp;
            }
            Ls[i] = Ls[i] + Vec(0.25f * clamp(L.x), 0.25f * clamp(L.y), 0.25f * clamp(L.z));
        }
    }
}

cudaError_t Render(Ray& cam, Vec* film, size_t w, size_t h, unsigned int spp = 100) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }
    const size_t n_pixels = w * h;
    Sphere* spheres_device;
    HandleError(cudaMalloc((void**)&spheres_device, sizeof(spheres)));
    HandleError(cudaMemcpy(spheres_device, spheres, sizeof(spheres), cudaMemcpyHostToDevice));
    Vec *film_device;
    cudaMalloc((void**)&film_device, sizeof(Vec) * n_pixels);
    cudaMemcpy(film_device, 0, sizeof(Vec) * n_pixels, cudaMemcpyHostToDevice);

    const dim3 nblocks(w / 16u, h / 16u);
    const dim3 nthreads(16u, 16u);
    render_kernel<<< nblocks, nthreads >>> (spheres_device, 4, cam, w, h, film_device, spp);
    
    HandleError(cudaMemcpy(film, film_device, n_pixels * sizeof(Vec), cudaMemcpyDeviceToHost));

    HandleError(cudaFree(spheres_device));
    HandleError(cudaFree(film_device));
    cudaDeviceSynchronize();
Error:
    return cudaStatus;
}

int main() {
    int w = 1024, h = 768, spp = 4; // # samples
    Ray cam(Vec(50.f, 52.f, 285.f), Vec(0.f, -0.042612f, -1.f).norm()); // cam pos, dir
    Vec* film = (Vec*)malloc(w * h * sizeof(Vec));
    Render(cam, film, w, h, spp);
    FILE* f = fopen("image.ppm", "w"); // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)
        fprintf(f, "%d %d %d ", toInt(film[i].x), toInt(film[i].y), toInt(film[i].z));
    free(film);
    return 0;
}
