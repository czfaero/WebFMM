export interface IKernel {
    debug: boolean;
    accelBuffer: Float32Array;
    Init: (particleBuffer: Float32Array) => Promise<void>;
    p2p: (numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any) => Promise<void>;
}
