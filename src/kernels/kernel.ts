export interface IKernel {
    Init: (particleBuffer: Float32Array) => Promise<void>;
    p2p: (numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any) => void
}
