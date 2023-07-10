import { FMMSolver } from '../FMMSolver';
import { IKernel } from './kernel';

const eps = 1e-6;
const inv4PI = 0.25 / Math.PI;

function complex_exp(re: number, im: number) {
    const tmp = Math.exp(re);
    return {
        re: Math.cos(im) * tmp,
        im: Math.sin(im) * tmp
    }
}

export class KernelTs implements IKernel {
    debug: boolean;
    core: FMMSolver;

    particleBuffer: Float32Array; // vec4
    particleCount: number;

    factorial: Float32Array;
    constructor(core: FMMSolver) {
        this.debug = false;
        this.core = core;
    }
    accelBuffer: Float32Array;
    async Init(particleBuffer: Float32Array) {
        this.particleCount = particleBuffer.length / 4;
        this.particleBuffer = particleBuffer;
        this.accelBuffer = new Float32Array(this.particleCount * 3);
        this.precalc();
    }
    Anm: Float32Array;
    anm: Float32Array;
    precalc() {
        const core = this.core;
        this.Anm = new Float32Array(core.numExpansion4);
        this.anm = new Float32Array(4 * core.numExpansion2);
        this.factorial = new Float32Array(4 * core.numExpansion2);
        //   int n, m, nm, nabsm, j, k, nk, npn, nmn, npm, nmm, nmk, i, nmk1, nm1k, nmk2;
        //         vec3 < int > boxIndex3D;
        //         vec3 < double > dist;
        //   double anmk[2][numExpansion4];
        //   double Dnmd[numExpansion4];
        //   double fnma, fnpa, pn, p, p1, p2, anmd, anmkd, rho, alpha, beta, sc, ank, ek;
        //         std:: complex < double > expBeta[numExpansion2], I(0.0, 1.0);

        //   int jk, jkn, jnk;
        //   double fnmm, fnpm, fad;

        for (let n = 0; n < 2 * core.numExpansions; n++) {
            for (let m = -n; m <= n; m++) {
                let nm = n * n + n + m;
                const nabsm = Math.abs(m);
                let fnmm = 1.0;
                for (let i = 1; i <= n - m; i++)
                    fnmm *= i;
                let fnpm = 1.0;
                for (let i = 1; i <= n + m; i++)
                    fnpm *= i;
                let fnma = 1.0;
                for (let i = 1; i <= n - nabsm; i++)
                    fnma *= i;
                let fnpa = 1.0;
                for (let i = 1; i <= n + nabsm; i++)
                    fnpa *= i;
                this.factorial[nm] = Math.sqrt(fnma / fnpa);
                const fad = Math.sqrt(fnmm * fnpm);
                this.anm[nm] = Math.pow(-1.0, n) / fad;
            }
        }

        for (let j = 0; j < core.numExpansions; j++) {
            for (let k = -j; k <= j; k++) {
                const jk = j * j + j + k;
                for (let n = Math.abs(k); n < core.numExpansions; n++) {
                    const nk = n * n + n + k;
                    const jkn = jk * core.numExpansion2 + nk;
                    const jnk = (j + n) * (j + n) + j + n;
                    this.Anm[jkn] = Math.pow(-1.0, j + k) * this.anm[nk] * this.anm[jk] / this.anm[jnk];
                }
            }
        }

        // let pn = 1;
        // for (let m = 0; m < 2 * this.core.numExpansions; m++) {
        //     let p = pn;
        //     let npn = m * m + 2 * m;
        //     let nmn = m * m;
        //     Ynm[npn] = factorial[npn] * p;
        //     Ynm[nmn] = conj(Ynm[npn]);
        //     let p1 = p;
        //     p = (2 * m + 1) * p;
        //     for (let n = m + 1; n < 2 * this.core.numExpansions; n++) {
        //         npm = n * n + n + m;
        //         nmm = n * n + n - m;
        //         Ynm[npm] = factorial[npm] * p;
        //         Ynm[nmm] = conj(Ynm[npm]);
        //         p2 = p1;
        //         p1 = p;
        //         p = ((2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
        //     }
        //     pn = 0;
        // }

        // for (let n = 0; n < core.numExpansions; n++) {
        //     for (let m = 1; m <= n; m++) {
        //         anmd = n * (n + 1) - m * (m - 1);
        //         for (k = 1 - m; k < m; k++) {
        //             nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //             anmkd = ((double)(n * (n + 1) - k * (k + 1))) / (n * (n + 1) - m * (m - 1));
        //             anmk[0][nmk] = -(m + k) / sqrt(anmd);
        //             anmk[1][nmk] = sqrt(anmkd);
        //         }
        //     }
        // }

        // for (let i = 0; i < numRelativeBox; i++) {
        //     tree.unmorton(i, boxIndex3D);
        //     dist.x = boxIndex3D.x - 3;
        //     dist.y = boxIndex3D.y - 3;
        //     dist.z = boxIndex3D.z - 3;
        //     cart2sph(rho, alpha, beta, dist.x, dist.y, dist.z);

        //     sc = sin(alpha) / (1 + cos(alpha));
        //     for (n = 0; n < 4 * numExpansions - 3; n++) {
        //         expBeta[n] = exp((n - 2 * numExpansions + 2) * beta * I);
        //     }

        //     for (n = 0; n < numExpansions; n++) {
        //         nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + n * (2 * n + 1) + n;
        //         Dnmd[nmk] = pow(cos(alpha * 0.5), 2 * n);
        //         for (k = n; k >= 1 - n; k--) {
        //             nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + n * (2 * n + 1) + k;
        //             nmk1 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + n * (2 * n + 1) + k - 1;
        //             ank = ((double)n + k) / (n - k + 1);
        //             Dnmd[nmk1] = sqrt(ank) * tan(alpha * 0.5) * Dnmd[nmk];
        //         }
        //         for (m = n; m >= 1; m--) {
        //             for (k = m - 1; k >= 1 - m; k--) {
        //                 nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //                 nmk1 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k + 1;
        //                 nm1k = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + (m - 1) * (2 * n + 1) + k;
        //                 Dnmd[nm1k] = anmk[1][nmk] * Dnmd[nmk1] + anmk[0][nmk] * sc * Dnmd[nmk];
        //             }
        //         }
        //     }

        //     for (n = 1; n < numExpansions; n++) {
        //         for (m = 0; m <= n; m++) {
        //             for (k = -m; k <= -1; k++) {
        //                 ek = pow(-1.0, k);
        //                 nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //                 nmk1 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 - k * (2 * n + 1) - m;
        //                 Dnmd[nmk] = ek * Dnmd[nmk];
        //                 Dnmd[nmk1] = pow(-1.0, m + k) * Dnmd[nmk];
        //             }
        //             for (k = 0; k <= m; k++) {
        //                 nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //                 nmk1 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + k * (2 * n + 1) + m;
        //                 nmk2 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 - k * (2 * n + 1) - m;
        //                 Dnmd[nmk1] = pow(-1.0, m + k) * Dnmd[nmk];
        //                 Dnmd[nmk2] = Dnmd[nmk1];
        //             }
        //         }
        //     }

        //     for (n = 0; n < numExpansions; n++) {
        //         for (m = 0; m <= n; m++) {
        //             for (k = -n; k <= n; k++) {
        //                 nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //                 nk = n * (n + 1) + k;
        //                 Dnm[i][m][nk] = Dnmd[nmk] * expBeta[k + m + 2 * numExpansions - 2];
        //             }
        //         }
        //     }

        //     alpha = -alpha;
        //     beta = -beta;

        //     sc = sin(alpha) / (1 + cos(alpha));
        //     for (n = 0; n < 4 * numExpansions - 3; n++) {
        //         expBeta[n] = exp((n - 2 * numExpansions + 2) * beta * I);
        //     }

        //     for (n = 0; n < numExpansions; n++) {
        //         nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + n * (2 * n + 1) + n;
        //         Dnmd[nmk] = pow(cos(alpha * 0.5), 2 * n);
        //         for (k = n; k >= 1 - n; k--) {
        //             nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + n * (2 * n + 1) + k;
        //             nmk1 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + n * (2 * n + 1) + k - 1;
        //             ank = ((double)n + k) / (n - k + 1);
        //             Dnmd[nmk1] = sqrt(ank) * tan(alpha * 0.5) * Dnmd[nmk];
        //         }
        //         for (m = n; m >= 1; m--) {
        //             for (k = m - 1; k >= 1 - m; k--) {
        //                 nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //                 nmk1 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k + 1;
        //                 nm1k = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + (m - 1) * (2 * n + 1) + k;
        //                 Dnmd[nm1k] = anmk[1][nmk] * Dnmd[nmk1] + anmk[0][nmk] * sc * Dnmd[nmk];
        //             }
        //         }
        //     }

        //     for (n = 1; n < numExpansions; n++) {
        //         for (m = 0; m <= n; m++) {
        //             for (k = -m; k <= -1; k++) {
        //                 ek = pow(-1.0, k);
        //                 nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //                 nmk1 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 - k * (2 * n + 1) - m;
        //                 Dnmd[nmk] = ek * Dnmd[nmk];
        //                 Dnmd[nmk1] = pow(-1.0, m + k) * Dnmd[nmk];
        //             }
        //             for (k = 0; k <= m; k++) {
        //                 nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //                 nmk1 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + k * (2 * n + 1) + m;
        //                 nmk2 = (4 * n * n * n + 6 * n * n + 5 * n) / 3 - k * (2 * n + 1) - m;
        //                 Dnmd[nmk1] = pow(-1.0, m + k) * Dnmd[nmk];
        //                 Dnmd[nmk2] = Dnmd[nmk1];
        //             }
        //         }
        //     }

        //     for (n = 0; n < numExpansions; n++) {
        //         for (m = 0; m <= n; m++) {
        //             for (k = -n; k <= n; k++) {
        //                 nmk = (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        //                 nk = n * (n + 1) + k;
        //                 Dnm[i + numRelativeBox][m][nk] = Dnmd[nmk] * expBeta[k + m + 2 * numExpansions - 2];
        //             }
        //         }
        //     }
        // }

        // for (j = 0; j < numBoxIndexTotal; j++) {
        //     for (i = 0; i < numCoefficients; i++) {
        //         Mnm[j][i] = 0;
        //     }
        // }
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


    cart2sph(dx: number, dy: number, dz: number) {
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz) + eps;
        const theta = Math.acos(dz / r);
        let phi: number;
        if (Math.abs(dx) + Math.abs(dy) < eps) {
            phi = 0;
        }
        else if (Math.abs(dx) < eps) {
            phi = dy / Math.abs(dy) * Math.PI * 0.5;
        }
        else if (dx > 0) {
            phi = Math.atan(dy / dx);
        }
        else {
            phi = Math.atan(dy / dx) + Math.PI;
        }
        return { rho: r, alpha: theta, beta: phi };
    }

    Mnm: Array<Float32Array>;
    async p2m(numBoxIndex: number, particleOffset: any) {
        const core = this.core;
        const particleBuffer = this.particleBuffer;

        this.Mnm = new Array(core.numBoxIndexTotal);// ?

        let YnmReal = new Float32Array(core.numExpansion2);


        const boxSize = core.rootBoxSize / (1 << core.maxLevel);
        for (let jj = 0; jj < numBoxIndex; jj++) {
            const boxIndex3D = core.unmorton(core.boxIndexFull[jj]);

            const boxCenterX = core.boxMinX + (boxIndex3D.x + 0.5) * boxSize;
            const boxCenterY = core.boxMinY + (boxIndex3D.y + 0.5) * boxSize;
            const boxCenterZ = core.boxMinZ + (boxIndex3D.z + 0.5) * boxSize;

            const MnmVec = new Float32Array(core.numCoefficients * 2);
            for (let j = particleOffset[0][jj]; j <= particleOffset[1][jj]; j++) {
                const dx = particleBuffer[j * 4] - boxCenterX;
                const dy = particleBuffer[j * 4 + 1] - boxCenterY;
                const dz = particleBuffer[j * 4 + 2] - boxCenterZ;
                const { rho, alpha, beta } = this.cart2sph(dx, dy, dz);
                const xx = Math.cos(alpha);
                const s2 = Math.sqrt((1 - xx) * (1 + xx));
                let fact = 1;
                let pn = 1;
                let rhom = 1;
                for (let m = 0; m < core.numExpansions; m++) {
                    let p = pn;
                    let nm = m * m + 2 * m;
                    YnmReal[nm] = rhom * this.factorial[nm] * p;
                    let p1 = p;
                    p = xx * (2 * m + 1) * p;
                    rhom *= rho;
                    let rhon = rhom;
                    for (let n = m + 1; n < core.numExpansions; n++) {
                        nm = n * n + n + m;
                        YnmReal[nm] = rhon * this.factorial[nm] * p;
                        let p2 = p1;
                        p1 = p;
                        p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
                        rhon *= rho;
                    }
                    pn = -pn * fact * s2;
                    fact += 2;
                }
                for (let n = 0; n < core.numExpansions; n++) {
                    for (let m = 0; m <= n; m++) {
                        let nm = n * n + n + m;
                        let nms = n * (n + 1) / 2 + m;
                        let eim = complex_exp(0, -m * beta);
                        const w = particleBuffer[j * 4 + 3];

                        MnmVec[nms * 2] += w * YnmReal[nm] * eim.re;
                        MnmVec[nms * 2 + 1] += w * YnmReal[nm] * eim.im;

                    }
                }
            }
            this.Mnm[jj] = MnmVec;
        }


    }

}