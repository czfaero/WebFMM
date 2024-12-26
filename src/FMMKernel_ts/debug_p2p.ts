

export function debug_p2p(core, src, dst) {
    const tree = core.tree;
    const dst_start = tree.nodeStartOffset[dst];
    const dst_count = tree.nodeEndOffset[dst] - dst_start + 1;
    const src_start = tree.nodeStartOffset[src];
    const src_count = tree.nodeEndOffset[src] - src_start + 1;

    let direct_result = new Float32Array(dst_count * 3);
    function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    function inverseSqrt(x) { return 1 / Math.sqrt(x); }
    const eps = 1e-6;
    for (let dst_i = 0; dst_i < dst_count; dst_i++) {
        let accel = { x: 0, y: 0, z: 0 };
        let dst_index = dst_start + dst_i;
        let dst_node = tree.getNode(dst_index);
        for (let src_i = 0; src_i < src_count; src_i++) {
            let src_index = src_start + src_i;
            if (dst_index == src_index) { continue; }
            let src_node = tree.getNode(src_index);
            let dist = {
                x: dst_node.x - src_node.x,
                y: dst_node.y - src_node.y,
                z: dst_node.z - src_node.z
            };
            let invDist = inverseSqrt(dot(dist, dist) + eps);
            let invDistCube = invDist * invDist * invDist;
            accel.x += invDistCube * dist.x;
            accel.y += invDistCube * dist.y;
            accel.z += invDistCube * dist.z;
        }
        direct_result[dst_i * 3] = accel.x;
        direct_result[dst_i * 3 + 1] = accel.y;
        direct_result[dst_i * 3 + 2] = accel.z;
    }
    return direct_result;
}