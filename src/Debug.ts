export enum DebugMode {
    off = 0,
    log = "log",
    debugger = "debugger",
    retry_immediate = "retry_immediate", // for RenderDoc, PIX
    retry_at_end = "retry_at_end"// 
}

export function debug_FindNaN(buffer) {
    let result = [];
    for (const [i, x] of buffer.entries()) {
        if (Number.isNaN(x)) {
            result.push(i);
        }
    }
    return result;
}


function RecordVideo() {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    // Optional frames per second argument.
    const stream = canvas.captureStream(30);
    const recordedChunks = [];

    console.log(stream);
    const options = {
        mimeType:
            //"video/webm; codecs=vp9" 
            "video/mp4", videoBitsPerSecond: 2500000,
    };
    const mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.ondataavailable = handleDataAvailable;
    mediaRecorder.start();

    function handleDataAvailable(event) {
        console.log("data-available");
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
            console.log(recordedChunks);
            download();
        } else {
            // …
        }
    }
    function download() {
        const blob = new Blob(recordedChunks, {
            type: "video/webm",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        document.body.appendChild(a);
        a.style.cssText = "display: none";
        a.href = url;
        a.download = "test.webm";
        a.click();
        window.URL.revokeObjectURL(url);
    }

    setTimeout((event) => {
        console.log("stopping");
        mediaRecorder.stop();
    }, 10000);

}