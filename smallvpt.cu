#include "cuda_runtime.h"               // A small gpu volumetric path tracer in 200 lines
#include "device_launch_parameters.h"   // Jerry Guo (c) CGV TU Delft
#include "math_constants.h"             // Based on smallvpt and cu-smallpt
#include "curand_kernel.h"              // Compile: nvcc
#include <stdlib.h>                     // Usage: cusmallvpt [#SPP]
#include <stdio.h>                      // Result: image.ppm

enum Refl_t { DIFF, SPEC, REFR }; 
inline void HandleError(cudaError_t err) {
    if (cudaSuccess != err) { printf("%s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}
struct Vec { // position, also color (r,g,b) 
    float x, y, z; 
    __host__ __device__ explicit Vec() { x = 0.f; y = 0.f; z = 0.f; }
    __host__ __device__ explicit Vec(float v) { x = v; y = v; z = v; }
    __host__ __device__ explicit Vec(float x_ = 0.f, float y_ = 0.f, float z_ = 0.f) { x = x_; y = y_; z = z_; }
    Vec(const Vec& vec) noexcept = default;
    Vec(Vec&& vec) noexcept = default;
    ~Vec() = default;
    __device__ Vec& operator=(const Vec& b) { this->x = b.x; this->y = b.y; this->z = b.z; return *this; }
    __device__ const Vec operator+(const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }
    __device__ const Vec operator-(const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ const Vec operator*(float b) const { return Vec(x * b, y * b, z * b); }
    __device__ const Vec mult(const Vec& b) const { return Vec(x * b.x, y * b.y, z * b.z); }
    __device__ float len() const { return sqrt(x * x + y * y + z * z); }
    __device__ Vec& norm() { float inv_len = 1.f / len(); this->x *= inv_len; this->y *= inv_len; this->z *= inv_len; return *this; }
    __device__ float dot(const Vec& b) const { return x * b.x + y * b.y + z * b.z; } // cross: 
    __device__ Vec operator%(Vec& b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
    __device__ Vec operator%(const Vec& b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};
__device__ inline float len(const Vec& v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
__device__ inline Vec norm(const Vec& v) { float inv_len = 1.f / len(v); return Vec(v.x * inv_len, v.y * inv_len, v.z * inv_len); }
struct Ray {
    Vec o, d;
    __host__ __device__ explicit Ray() : o(Vec(0.f, 0.f, 0.f)), d(Vec(0.f, 0.f, 0.f)) {}
    __host__ __device__ explicit Ray(Vec o_, Vec d_) noexcept : o(o_), d(d_) {}
    Ray(const Ray& ray) noexcept = default;
    Ray(Ray&& ray) noexcept = default;
    ~Ray() = default;
    __device__ Ray& operator=(const Ray& r) { this->o = r.o; this->d = r.d; return *this; }
};
struct Sphere {
    float rad;       
    Vec p, e, c;     
    Refl_t refl;     
    __host__ __device__ explicit Sphere(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
    __device__ float intersect(const Ray& r, float* tin = NULL, float* tout = NULL) const { 
        Vec op = p - r.o; 
        float t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
        if (det < 0.f) return 0; else det = sqrt(det);
        if (tin && tout) { *tin = (b - det <= 0.f) ? 0.f : b - det; *tout = b + det; }
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0.f);
    }
};
__host__ __device__ inline float clamp(float x) { return x < 0.f ? 0.f : x>1.f ? 1.f : x; }
__host__ __device__ inline int toInt(float x) { return int(pow(clamp(x), 1.f / 2.2f) * 255.f + .5f); }
__device__ inline bool intersect(const Sphere* spheres, size_t n_sphere, const Ray& r, float& t, int& id, float tmax = 1e20) {
    float d, inf = t = tmax;
    for (int i = int(n_sphere); i--;) if ((d = spheres[i].intersect(r)) && d < t) { t = d; id = i; }
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
    float s = 1.f-2.f*e1,cost=(s+2.f*g*g*g*(-1.0+e1)*e1+g*g*s+2.f*g*(1.f-e1+e1*e1))/((1.f+g*s)*(1.f+g*s)),sint=sqrt(1.f-cost*cost);
    return Vec(cos(2.f * CUDART_PI_F * e2) * sint, sin(2.f * CUDART_PI_F * e2) * sint, cost);
}
__device__ inline void generateOrthoBasis(Vec& u, Vec& v, Vec w) {
    Vec coVec = w;
    if (fabs(w.x) <= fabs(w.y))
        if (fabs(w.x) <= fabs(w.z)) coVec = Vec(0.f, -w.z, w.y); else coVec = Vec(-w.y, w.x, 0.f);
    else if (fabs(w.y) <= fabs(w.z)) coVec = Vec(-w.z, 0.f, w.x); else coVec = Vec(-w.y, w.x, 0.f);
    coVec.norm(); u = w % coVec, v = w % u;
}
__device__ inline float scatter(const Ray& r, Ray* sRay, float tin, float tout, float& s, const float& sigma_s, curandState_t* rand_state) {
    s = sampleSegment(curand_uniform(rand_state), sigma_s, tout - tin);
    Vec x = r.o + r.d * tin + r.d * s;
    Vec dir = sampleHG(-0.5f, curand_uniform(rand_state), curand_uniform(rand_state)); 
    Vec u(0.f, 0.f, 0.f), v(0.f, 0.f, 0.f);
    generateOrthoBasis(u, v, r.d);
    dir = u * dir.x + v * dir.y + r.d * dir.z;
    if (sRay)	*sRay = Ray(x, dir);
    return (1.0f - exp(-sigma_s * (tout - tin)));
}
__device__ Vec radiance(const Sphere* spheres, size_t n_sphere, const Ray& r, int _depth, curandState_t* rand_state) {
    Ray ray = r;
    Vec L(0.f, 0.f, 0.f);
    Vec B(1.f, 1.f, 1.f);
    int depth = _depth;
    float tnear, tfar, scaleBy = 1.f, absorption = 1.f;
    const Sphere homoMedium(300.f, Vec(50.f, 50.f, 80.f), Vec(0.f, 0.f, 0.f), Vec(0.f, 0.f, 0.f), DIFF);
    const float sigma_s = 0.009f, sigma_a = 0.006f, sigma_t = sigma_s + sigma_a;
    while (1) {
        float t;                               // distance to intersection
        int id = 0;                               // id of intersected object
        if (homoMedium.intersect(ray, &tnear, &tfar) > 0) {
            Ray sRay;
            float s, ms = scatter(ray, &sRay, tnear, tfar, s, sigma_s, rand_state), prob_s = ms;
            scaleBy = 1.f / (1.f - prob_s);
            if (curand_uniform(rand_state) <= prob_s) {// Sample surface or volume?
                if (!intersect(spheres, n_sphere, ray, t, id, tnear + s)) {
                    B = B * ms * (1.f - prob_s); ray = sRay; ++depth; continue;
                }
                scaleBy = 1.f;
            } else if (!intersect(spheres, n_sphere, ray, t, id)) return L;
            if (t >= tnear) {
                float dist = (t > tfar ? tfar - tnear : t - tnear);
                absorption = exp(-sigma_t * dist);
            }
        } else if (!intersect(spheres, n_sphere, ray, t, id)) return L;
        const Sphere& obj = spheres[id]; 
        Vec x = r.o + r.d * t, n = Vec(x - obj.p).norm(), nl = n.dot(ray.d) < 0 ? n : n * -1, f = obj.c, Le = obj.e;
        float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; 
        if (++depth > 5) if (curand_uniform(rand_state) < p) B = B * (1 / p); else  return L;
        if (n.dot(nl) > 0 || obj.refl != REFR) { B = B * absorption; Le = obj.e * absorption; } else scaleBy = 1.f;
        // Accumulate luminance and throughtput
        L = L + B.mult(Le); B = B.mult(f * scaleBy); ++depth;
        switch (obj.refl) {
            case SPEC: { ray = Ray(x, r.d - n * 2 * n.dot(r.d)); break; }
            case REFR: { 
                ray = Ray(x, r.d - n * 2 * n.dot(r.d)); bool into = n.dot(nl) > 0; 
                float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
                if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) break;
                Vec tdir = Vec(r.d*nnt-n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
                float a=nt-nc,b=nt+nc,R0=a*a/(b*b),c = 1 - (into ? -ddn : tdir.dot(n));
                float Re=R0+(1-R0)*c*c*c*c*c, Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP = Tr / (1 - P);
                if (curand_uniform(rand_state) < P) B=B*RP; else { ray=Ray(x,tdir); B=B*TP; }
                break;
            }
            default: { 
                float r1=2*CUDART_PI_F*curand_uniform(rand_state),r2=curand_uniform(rand_state),r2s = sqrt(r2);
                Vec w = nl, u = Vec((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1.f, 1.f, 1.f)) % w).norm(), v = w % u;
                Vec d = Vec(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
                ray = Ray(x, d);
            }
        }
    }
}
__global__ void render_kernel(const Sphere* spheres, const size_t n_sphere, Vec* Ls, size_t w, size_t h, int spp) {
    const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t offset = x + y * blockDim.x * gridDim.x;
    const float inv_spp = 1.0f / float(spp);
    if (x >= w || y >= h) return;
    curandState rand_state; curand_init(offset, 0u, 0u, &rand_state);
    Ray cam(Vec(50.f, 52.f, 285.f), norm(Vec(0.f, -0.042612f, -1.f)));
    const float fov = 0.5135f; Vec cx = Vec(w * fov / h, 0.0f, 0.0f);
    Vec cy = norm(Vec(cx % cam.d)) * fov; size_t i = (h - 1u - y) * w + x;
    for (size_t sy = 0u; sy < 2u; ++sy) for (size_t sx = 0u; sx < 2u; ++sx) {
        Vec L(0.f, 0.f, 0.f);
        for (size_t s = 0u; s < spp; ++s) {
            float u1 = 2.f * curand_uniform(&rand_state);
            float u2 = 2.f * curand_uniform(&rand_state);
            float dx = (u1 < 1.f) ? sqrt(u1) - 1.f : 1.f - sqrt(2.f - u1);
            float dy = (u2 < 1.f) ? sqrt(u2) - 1.f : 1.f - sqrt(2.f - u2);
            Vec d = cx * (((sx+0.5+dx)*0.5+x)/w-0.5)+cy*(((sy+0.5+dy)*0.5+y)/h-0.5)+cam.d;
            Ray pRay(cam.o + d * 140.f, d.norm());
            L = L + radiance(spheres, n_sphere, pRay, 0, &rand_state) * inv_spp;
        }
        Ls[i] = Ls[i] + Vec(0.25f * clamp(L.x), 0.25f * clamp(L.y), 0.25f * clamp(L.z));
    }
}
cudaError_t Render(int w, int h, unsigned int spp = 100) {
    const size_t n_sphere = 4;
    Sphere spheres[n_sphere] = {//Scene: radius, position, emission, color, material 
        Sphere(26.5f, Vec(27.f, 18.5f, 78.f),Vec(0.f, 0.f, 0.f),Vec(1.f,1.f,1.f)*.75f,SPEC),//Mirr
        Sphere(12.f, Vec(70.f, 43.f, 78.f), Vec(0.f, 0.f, 0.f), Vec(0.27f,0.8f,0.8f), REFR),//Glas
        Sphere(8.f, Vec(55.f, 87.f, 78.f), Vec(0.f, 0.f, 0.f),  Vec(1,1,1) * .75f, DIFF), //Lite
        Sphere(4.f, Vec(55.f, 80.f, 78.f), Vec(10.f,10.f,10.f),  Vec(0.f, 0.f, 0.f), DIFF) //Lite
    };
    HandleError(cudaSetDevice(0));
    const size_t n_pixels = size_t(w * h);
    Sphere* spheres_device;
    HandleError(cudaMalloc((void**)&spheres_device, sizeof(spheres)));
    HandleError(cudaMemcpy(spheres_device, spheres, sizeof(spheres), cudaMemcpyHostToDevice));
    Vec* film_device;
    HandleError(cudaMalloc((void**)&film_device, sizeof(Vec) * n_pixels));
    HandleError(cudaMemset(film_device, 0, sizeof(Vec) * n_pixels));
    const dim3 nblocks(w / 16, h / 16);
    const dim3 nthreads(16, 16);
    render_kernel <<< nblocks, nthreads >>> (spheres_device, n_sphere, film_device, w, h, spp);
    Vec* film = (Vec*)malloc(n_pixels * sizeof(Vec));
    HandleError(cudaMemcpy(film, film_device, sizeof(Vec) * n_pixels, cudaMemcpyDeviceToHost));
    HandleError(cudaFree(spheres_device));
    HandleError(cudaFree(film_device));
    FILE* f = fopen("image.ppm", "w"); // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i=0;i<w*h;i++) fprintf(f,"%d %d %d ",toInt(film[i].x),toInt(film[i].y),toInt(film[i].z));
    free(film); return cudaSuccess;
}
int main(int argc, char* argv[]) {
    int w = 1024, h = 768, spp = argc == 2 ? atoi(argv[1]) / 4 : 100; 
    Render(w, h, spp); return 0;
}
