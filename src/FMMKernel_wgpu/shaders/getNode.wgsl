fn getNode(i : u32) -> vec4f{
    return vec4f(
        nodeBuffer[i * 4],
        nodeBuffer[i * 4 + 1],
        nodeBuffer[i * 4 + 2],
        nodeBuffer[i * 4 + 3]
    );
}