export enum DebugMode {
    off = 0,
    log = "log",
    debugger = "debugger"
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
