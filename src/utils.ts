const sqrt = Math.sqrt;
const abs = Math.abs;
const pow = Math.pow;
const acos = Math.acos;
const atan = Math.atan;
const cos = Math.cos;
const sin = Math.sin;
const sign=Math.sign;
const PI = Math.PI;

class vec3f {
    x: number;
    y: number;
    z: number;
}

/**
 * morton
 * @param boxIndex 
 * @returns object x y z 
 */
export function GetIndex3D(boxIndex: number): vec3f {
    var mortonIndex3D = [0, 0, 0];

    var n = boxIndex;
    var k = 0;
    var i = 0;
    while (n != 0) {
        let j = 2 - k;
        mortonIndex3D[j] += (n % 2) * (1 << i);
        n >>= 1;
        k = (k + 1) % 3;
        if (k == 0) { i++; }
    }
    return {
        x: mortonIndex3D[1],
        y: mortonIndex3D[2],
        z: mortonIndex3D[0]
    };
}
/**
 * morton index
 * @param boxIndex3D x,y,z
 * @param numLevel 
 * @returns 
 */
export function GetIndexFrom3D(boxIndex3D, numLevel: number) {

    let boxIndex = 0;
    for (let i = 0; i < numLevel; i++) {
        let nx = boxIndex3D.x % 2;
        boxIndex3D.x >>= 1;
        boxIndex += nx * (1 << (3 * i + 1));

        let ny = boxIndex3D.y % 2;
        boxIndex3D.y >>= 1;
        boxIndex += ny * (1 << (3 * i));

        let nz = boxIndex3D.z % 2;
        boxIndex3D.z >>= 1;
        boxIndex += nz * (1 << (3 * i + 2));
    }
    return boxIndex
}

/**
 * 
 * @param d 
 * @returns (r, theta, phi); 0 <= theta <= PI
 */
export function cart2sph(d) {
    const eps = 1e-6;
    var r = sqrt(d.x * d.x + d.y * d.y + d.z * d.z) + eps;
    var theta = acos(d.z / r);
    var phi;
    if (abs(d.x) + abs(d.y) < eps) {
        phi = 0;
    }
    else if (abs(d.x) < eps) {
        phi = sign(d.y) * PI * 0.5;
    }
    else if (d.x > 0) {
        phi = atan(d.y / d.x);
    }
    else {
        phi = atan(d.y / d.x) + PI;
    }
    function vec3f(x, y, z) { return { x: x, y: y, z: z }; }
    return vec3f(r, theta, phi);
}