let rawData = null;
let vertex: Float32Array = null;
let link: Uint32Array = null;
async function LoadData() {
    if (rawData) return;
    rawData = await (await fetch("/init.bin")).arrayBuffer();
    vertex = new Float32Array(rawData);
    console.log("vertex bin size:" + vertex.length)
    console.log(`[0] ${Array.from(vertex.subarray(0, 4)).map(x => x.toFixed(2)).join(' ')}`)


    link = new Uint32Array(10);
    for (let i = 0; i < link.length; i++) {
        link[i] = Math.floor((i + 1) / 2);
       
    }
    console.log(link)
}

export const GetNodes = async function (count: number) {
    await LoadData();
    let buf = new Float32Array(count * 4);
    for (let i = 0; i < count * 4; i++) {
        buf[i] = vertex[i];
    }
    return buf;
}
export const GetLinks = async function () {
    await LoadData();

    return link;
}

