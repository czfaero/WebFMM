const size_a = 50, size_b = 50, size_c = 50;
const interval = 0.5;
const count = size_a * size_b * size_c;
export const GetNodes = function () {

    const baseX = -Math.floor(size_a / 2) * interval,
        baseY = -Math.floor(size_b / 2) * interval,
        baseZ = -Math.floor(size_c / 2) * interval;
    const iterable = (function* () {

        for (let x = 0; x < size_a; x++)
            for (let y = 0; y < size_b; y++)
                for (let z = 0; z < size_c; z++) {
                    yield x * interval + baseX;
                    yield y * interval + baseY;
                    yield z * interval + baseZ;
                    yield 1;
                }
    })();
    const nodes = new Float32Array(iterable);
    //console.log(nodes.length);
    return nodes;
}
export const GetLinks = function () {
    const iterable = (function* () {
        for (let x = 0; x < size_a; x++)
            for (let y = 0; y < size_b; y++)
                for (let z = 0; z < size_c - 1; z++) {
                    let i = x * size_b * size_c + y * size_c + z;
                    yield i;
                    yield i + 1;
                }
        for (let x = 0; x < size_a; x++)
            for (let y = 0; y < size_b - 1; y++) {
                let i = x * size_b * size_c + y * size_c;
                yield i;
                yield i + size_c;
            }
        for (let x = 0; x < size_a - 1; x++) {
            let i = x * size_b * size_c;
            yield i;
            yield i + size_b * size_c;
        }

    })();


    const links = new Uint32Array(iterable);

    return links;
}
export const GetNodeColors = function () {
    let r = 1, g = 0, b = 0, t = 0;
    const iterable = (function* () {
        for (let x = 0; x < size_a; x++)
            for (let y = 0; y < size_b; y++)
                for (let z = 0; z < size_c; z++) {
                    yield r;
                    yield g;
                    yield b;
                    t = g;
                    g = r;
                    r = b;
                    b = t;
                }
    })();
    const colors = new Float32Array(iterable);
    console.log(colors.length);
    return colors;
}