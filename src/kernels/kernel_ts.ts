import { IKernel } from './kernel';

const eps = 1e-6;
const inv4PI = 0.25 / Math.PI;

export class KernelTs implements IKernel {
    debug: boolean;

    particleBuffer: Float32Array; // vec4
    particleCount: number;
    constructor() {
        this.debug = false;
    }
    accelBuffer: Float32Array;
    async Init(particleBuffer: Float32Array) {
        this.particleCount = particleBuffer.length / 4;
        this.particleBuffer = particleBuffer;
        this.accelBuffer = new Float32Array(this.particleCount * 3);
    }
    async p2p(numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any) {

        for (let ii = 0; ii < numBoxIndex; ii++) {
            for (let ij = 0; ij < numInteraction[ii]; ij++) {
                const jj = interactionList[ii][ij];
                for (let i = particleOffset[0][ii]; i <= particleOffset[1][ii]; i++) {
                    let ax = 0, ay = 0, az = 0;
                    for (let j = particleOffset[0][jj]; j <= particleOffset[1][jj]; j++) {
                        const dx = this.particleBuffer[i * 4] - this.particleBuffer[j * 4];
                        const dy = this.particleBuffer[i * 4 + 1] - this.particleBuffer[j * 4 + 1];
                        const dz = this.particleBuffer[i * 4 + 2] - this.particleBuffer[j * 4 + 2];
                        const invDist = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz + eps);
                        const invDistCude = invDist * invDist * invDist;
                        const s = this.particleBuffer[j * 4 + 3] * invDistCude;
                        ax -= dx * s * inv4PI;
                        ay -= dy * s * inv4PI;
                        az -= dz * s * inv4PI;
                    }
                    this.accelBuffer[i * 3] += ax;
                    this.accelBuffer[i * 3 + 1] += ay;
                    this.accelBuffer[i * 3 + 2] += az;
                }
            }
        }

    }

}