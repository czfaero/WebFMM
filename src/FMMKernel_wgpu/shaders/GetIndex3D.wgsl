fn GetIndex3D(boxIndex: u32) -> vec3<u32> {
    var mortonIndex3D: array<u32,3>;
    mortonIndex3D[0] = 0;
    mortonIndex3D[1] = 0;
    mortonIndex3D[2] = 0;
    var n = boxIndex;
    var k = 0;
    var i: u32 = 0;
    while (n != 0) {
        let j = 2 - k;
        mortonIndex3D[j] += (n % 2) * u32(1 << i);
        n >>= 1;
        k = (k + 1) % 3;
        if (k == 0) { i++; }
    }
    return vec3<u32>(
        mortonIndex3D[1],
        mortonIndex3D[2],
        mortonIndex3D[0]
    );
}