# Code from https://raytracing.github.io/books/RayTracingInOneWeekend.html

class Vec3:
    x : float
    y : float
    z : float

class Sphere:
    center : Vec3
    radius : float

class Ray:
    org : Vec3
    dir : Vec3

def make_vec3(x : In[float], y : In[float], z : In[float]) -> Vec3:
    ret : Vec3
    ret.x = x
    ret.y = y
    ret.z = z
    return ret

def add(a : In[Vec3], b : In[Vec3]) -> Vec3:
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z)

def sub(a : In[Vec3], b : In[Vec3]) -> Vec3:
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z)

def mul(a : In[float], b : In[Vec3]) -> Vec3:
    return make_vec3(a * b.x, a * b.y, a * b.z)

def dot(a : In[Vec3], b : In[Vec3]) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z

def normalize(v : In[Vec3]) -> Vec3:
    l : float = sqrt(dot(v, v))
    return make_vec3(v.x / l, v.y / l, v.z / l)

# Returns distance. If distance is zero or negative, the hit misses
def sphere_isect(sph : In[Sphere], ray : In[Ray]) -> float:
    oc : Vec3 = sub(ray.org, sph.center)
    a : float = dot(ray.dir, ray.dir)
    b : float = 2 * dot(oc, ray.dir)
    c : float = dot(oc, oc) - sph.radius * sph.radius
    discriminant : float = b * b - 4 * a * c
    ret_dist : float = 0
    if discriminant < 0:
        ret_dist = -1
    else:
        ret_dist = (-b - sqrt(discriminant)) / (2 * a)
    return ret_dist

def ray_color(ray : In[Ray], sph:In[Sphere]) -> Vec3:
    # sph : Sphere
    # sph.center = make_vec3(0, 0, -1)
    # sph.radius = 0.5

    ret_color : Vec3
    t : float = sphere_isect(sph, ray)

    N : Vec3
    white : Vec3 = make_vec3(1.0, 1.0, 1.0)
    blue : Vec3 = make_vec3(0.5, 0.7, 1.0)
    a : float

    if t > 0.0:
        N = normalize(sub(add(ray.org, mul(t, ray.dir)), sph.center))
        ret_color = make_vec3(0.5 * (N.x + 1.0), 0.5 * (N.y + 1.0), 0.5 * (N.z + 1.0))
    else:
        a = 0.5 * ray.dir.y + 1.0
        ret_color = add(mul((1.0 - a), white), mul(a, blue))
    return ret_color

def ray_color_constant(ray : In[Ray], sph:In[Sphere]) -> Vec3:
    # sph : Sphere
    # sph.center = make_vec3(0, 0, -1)
    # sph.radius = 0.5

    ret_color : Vec3
    t : float = sphere_isect(sph, ray)

    N : Vec3
    white : Vec3 = make_vec3(1, 1, 1)
    blue : Vec3 = make_vec3(0.5, 0.7, 1)
    black: Vec3 = make_vec3(0, 0, 0)
    red : Vec3 = make_vec3(0.5, 0, 0)
    a : float

    if t > 0:
        ret_color = red
    else:
        ret_color = black
    return ret_color


d_ray_color = rev_diff(ray_color_constant)

def clamp_int(x: In[int], min: In[int], max: In[int]) -> int:
    if x < min:
        return min
    else:
        if x > max:
            return max
        else:
            return x

def post_process(w: In[int], h: In[int], image: In[Array[Vec3]], output_img: Out[Array[Vec3]]):
    y: int = 0
    x: int
    left: int
    up: int
    pixel_center: Vec3
    intp_x: float
    intp_y: float
    intp_z: float
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            left = clamp_int(x - 1, 0, w)
            up = clamp_int(y - 1, 0, h)
            intp_x = 0.25 * (
                        image[w * y + x].x + image[w * up + x].x + image[w * y + left].x + image[w * up + left].x)
            intp_y = 0.25 * (
                        image[w * y + x].y + image[w * up + x].y + image[w * y + left].y + image[w * up + left].y)
            intp_z = 0.25 * (
                        image[w * y + x].z + image[w * up + x].z + image[w * y + left].z + image[w * up + left].z)
            output_img[w * y + x] = make_vec3(intp_x, intp_y, intp_z)
            x = x + 1
        y = y + 1

def d_post_process(w: In[int], dw: Out[int], h: In[int], dh: Out[int], image: In[Array[Vec3]], d_image: Out[Array[Vec3]], d_output_img: In[Array[Vec3]]):
    y: int = 0
    x: int
    left: int
    up: int
    pixel_center: Vec3
    intp_x: float
    intp_y: float
    intp_z: float
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            left = clamp_int(x - 1, 0, w)
            up = clamp_int(y - 1, 0, h)
            d_image[w * up + x].x = d_image[w * up + x].x + d_output_img[w * y + x].x
            d_image[w * up + x].y = d_image[w * up + x].y + d_output_img[w * y + x].y
            d_image[w * up + x].z = d_image[w * up + x].z + d_output_img[w * y + x].z
            #
            d_image[w * up + left].x = d_image[w * up + left].x + d_output_img[w * y + x].x
            d_image[w * up + left].y = d_image[w * up + left].y + d_output_img[w * y + x].y
            d_image[w * up + left].z = d_image[w * up + left].z + d_output_img[w * y + x].z
            #
            d_image[w * y + x].x = d_image[w * y + x].x + d_output_img[w * y + x].x
            d_image[w * y + x].y = d_image[w * y + x].y + d_output_img[w * y + x].y
            d_image[w * y + x].z = d_image[w * y + x].z + d_output_img[w * y + x].z
            #
            d_image[w * y + left].x = d_image[w * y + left].x + d_output_img[w * y + x].x
            d_image[w * y + left].y = d_image[w * y + left].y + d_output_img[w * y + x].y
            d_image[w * y + left].z = d_image[w * y + left].z + d_output_img[w * y + x].z
            x = x + 1
        y = y + 1


def diff_raytrace(w : In[int], h : In[int], primal_image: In[Array[Vec3]], dl_dimage : In[Array[Vec3]], radius : In[float], dl_dpimage: In[Array[Vec3]]) -> float:
    # Camera setup
    aspect_ratio : float = int2float(w) / int2float(h)
    focal_length : float = 1.0
    viewport_height : float = 2.0
    viewport_width : float = viewport_height * aspect_ratio
    camera_center : Vec3 = make_vec3(0, 0, 0)
    # Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u : Vec3 = make_vec3(viewport_width / w, 0, 0)
    pixel_delta_v : Vec3 = make_vec3(0, -viewport_height / h, 0)
    # Calculate the location of the upper left pixel.
    viewport_upper_left : Vec3 = make_vec3(\
            camera_center.x - viewport_width / 2,
            camera_center.y + viewport_height / 2,
            camera_center.z - focal_length
        )
    pixel00_loc : Vec3 = viewport_upper_left
    pixel00_loc.x = pixel00_loc.x + pixel_delta_u.x / 2
    pixel00_loc.y = pixel00_loc.y - pixel_delta_v.y / 2

    y : int = 0
    x : int
    pixel_center : Vec3

    sphere : Sphere
    sphere.center = make_vec3(0, 0, -1)
    sphere.radius = radius

    dl_dsphere: Sphere

    ray: Ray
    d_ray: Ray

    dw: int
    dh: int

    d_post_process(w, dw, h, dh, primal_image, dl_dpimage, dl_dimage)

    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            pixel_center = add(add(pixel00_loc, mul(x, pixel_delta_u)), mul(y, pixel_delta_v))

            ray.org = make_vec3(camera_center.x, camera_center.y, camera_center.z)
            ray.dir = normalize(sub(pixel_center, camera_center))

            d_ray_color(ray, d_ray, sphere, dl_dsphere,  dl_dpimage[w * y + x])

            x = x + 1
        y = y + 1

    return dl_dsphere.radius



def raytrace(w : In[int], h : In[int], image : Out[Array[Vec3]], radius : In[float], tmp_img: In[Array[Vec3]]):
    # Camera setup
    aspect_ratio: float = int2float(w) / int2float(h)
    focal_length: float = 1.0
    viewport_height: float = 2.0
    viewport_width: float = viewport_height * aspect_ratio
    camera_center: Vec3 = make_vec3(0, 0, 0)
    # Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u: Vec3 = make_vec3(viewport_width / w, 0, 0)
    pixel_delta_v: Vec3 = make_vec3(0, -viewport_height / h, 0)
    # Calculate the location of the upper left pixel.
    viewport_upper_left: Vec3 = make_vec3( \
        camera_center.x - viewport_width / 2,
        camera_center.y + viewport_height / 2,
        camera_center.z - focal_length
    )
    pixel00_loc: Vec3 = viewport_upper_left
    pixel00_loc.x = pixel00_loc.x + pixel_delta_u.x / 2
    pixel00_loc.y = pixel00_loc.y - pixel_delta_v.y / 2

    sph: Sphere
    sph.radius = radius
    sph.center = make_vec3(0, 0, -1)

    y: int = 0
    x: int
    pixel_center: Vec3
    ray: Ray
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            pixel_center = add(add(pixel00_loc, mul(x, pixel_delta_u)), mul(y, pixel_delta_v))
            ray.org = camera_center
            ray.dir = normalize(sub(pixel_center, camera_center))
            tmp_img[w * y + x] = ray_color_constant(ray, sph)
            x = x + 1
        y = y + 1

    post_process(w, h, tmp_img, image)
