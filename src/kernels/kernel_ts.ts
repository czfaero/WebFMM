import { FMMSolver } from '../FMMSolver';
import { IKernel } from './kernel';

const eps = 1e-6;
const inv4PI = 0.25 / Math.PI;

const numRelativeBox = 512;        // max of relative box positioning

class Complex {
    re: number;
    im: number;
    multiply(cn2: Complex): Complex {
        const cn1 = this;
        return new Complex(
            cn1.re * cn2.re - cn1.im * cn2.im,
            cn1.re * cn2.im + cn1.im * cn2.re);
    }
    multiplyReal(x: number): Complex {
        const cn1 = this;
        return new Complex(
            cn1.re * x,
            cn1.im * x);
    }
    conj() {
        return new Complex(this.re, -this.im);
    }
    exp() {
        const tmp = Math.exp(this.re);
        return new Complex(
            Math.cos(this.im) * tmp,
            Math.sin(this.im) * tmp
        );
    }

    static fromBuffer(b: Float32Array, i: number): Complex {
        return new Complex(b[i * 2], b[i * 2 + 1]);
    };
    constructor(re: number, im = 0) {
        this.re = re;
        this.im = im;
    }
}


export class KernelTs implements IKernel {
    debug: boolean;
    core: FMMSolver;

    particleBuffer: Float32Array; // vec4
    particleCount: number;


    constructor(core: FMMSolver) {
        this.debug = false;
        this.core = core;
    }
    accelBuffer: Float32Array;
    async Init(particleBuffer: Float32Array) {
        this.particleCount = particleBuffer.length / 4;
        this.particleBuffer = particleBuffer;
        this.accelBuffer = new Float32Array(this.particleCount * 3);
        await this.precalc();
    }
    /** [4 * numExpansion2]; for p2m */
    factorial: Float32Array;
    /** complex [numBoxIndexLeaf][numCoefficients]; for  */
    Lnm: Array<Float32Array>;
    /** complex [numBoxIndexLeaf][numCoefficients]; for  */
    LnmOld: Array<Float32Array>;
    /** complex [numBoxIndexTotal][numCoefficients]; for  */
    Mnm: Array<Float32Array>;
    /** spherical harmonic; complex [4 * numExpansion2]; for m2m */
    Ynm: Float32Array;
    /** complex [2 * numRelativeBox][numExpansions][numExpansion2]; for rotation -> m2m */
    Dnm: Array<Array<Float32Array>>;

    // -----
    /** [numExpansion4]; for m2l */
    Anm: Float32Array;
    /** [4 * numExpansion2]; for m2m l2l*/
    anm: Float32Array;

    //debug
    Dnmd: Float32Array;
    expBeta: Float32Array;
    async precalc() {
        const core = this.core;
        this.Anm = new Float32Array(core.numExpansion4);
        this.anm = new Float32Array(4 * core.numExpansion2);
        this.factorial = new Float32Array(4 * core.numExpansion2);
        this.Lnm = new Array(core.numBoxIndexLeaf);
        for (let i = 0; i < core.numBoxIndexLeaf; i++) {
            this.Lnm[i] = new Float32Array(core.numCoefficients * 2);
        }
        this.LnmOld = new Array(core.numBoxIndexLeaf);
        this.Mnm = new Array(core.numBoxIndexTotal);
        this.Ynm = new Float32Array(4 * core.numExpansion2 * 2);
        this.Dnm = new Array(2 * numRelativeBox);
        for (let i = 0; i < 2 * numRelativeBox; i++) {
            this.Dnm[i] = new Array(core.numExpansions);
            for (let j = 0; j < core.numExpansions; j++)
                this.Dnm[i][j] = new Float32Array(core.numExpansion2 * 2);
        }
        /** [2][numExpansion4]; for Dnmd -> Dnm */
        const anmk = [new Float32Array(core.numExpansion4), new Float32Array(core.numExpansion4)];
        /** [numExpansion4]; for Dnm */
        const Dnmd = new Float32Array(core.numExpansion4);
        /** complex [numExpansion2]; for Dnm */
        const expBeta = new Float32Array(core.numExpansion2 * 2);

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

        let pn = 1;
        for (let m = 0; m < 2 * this.core.numExpansions; m++) {
            let p = pn;
            let npn = m * m + 2 * m;
            let nmn = m * m;
            this.Ynm[npn * 2] = this.factorial[npn] * p;
            this.Ynm[nmn * 2] = this.Ynm[npn * 2];//conj(Ynm[npn*2])
            let p1 = p;
            p = (2 * m + 1) * p;
            for (let n = m + 1; n < 2 * this.core.numExpansions; n++) {
                let npm = n * n + n + m;
                let nmm = n * n + n - m;
                this.Ynm[npm * 2] = this.factorial[npm] * p;
                this.Ynm[nmm * 2] = this.Ynm[npm * 2];//conj(Ynm[npm*2]);
                let p2 = p1;
                p1 = p;
                p = ((2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
            }
            pn = 0;
        }

        for (let n = 0; n < core.numExpansions; n++) {
            for (let m = 1; m <= n; m++) {
                let anmd = n * (n + 1) - m * (m - 1);
                for (let k = 1 - m; k < m; k++) {
                    let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                    let anmkd = ((n * (n + 1) - k * (k + 1))) / (n * (n + 1) - m * (m - 1));// double
                    anmk[0][nmk] = -(m + k) / Math.sqrt(anmd);
                    anmk[1][nmk] = Math.sqrt(anmkd);
                }
            }
        }

        for (let i = 0; i < numRelativeBox; i++) {
            let boxIndex3D = core.unmorton(i);
            let dx = boxIndex3D.x - 3;
            let dy = boxIndex3D.y - 3;
            let dz = boxIndex3D.z - 3;
            let { rho, alpha, beta } = this.cart2sph(dx, dy, dz);


            let sc = Math.sin(alpha) / (1 + Math.cos(alpha));
            for (let n = 0; n < 4 * core.numExpansions - 3; n++) {
                let c = new Complex(0, (n - 2 * core.numExpansions + 2) * beta);
                let c2 = c.exp();
                expBeta[n * 2] = c2.re;
                expBeta[n * 2 + 1] = c2.im;
            }

            for (let n = 0; n < core.numExpansions; n++) {
                let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + n * (2 * n + 1) + n;
                Dnmd[nmk] = Math.pow(Math.cos(alpha * 0.5), 2 * n);


                for (let k = n; k >= 1 - n; k--) {
                    nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + n * (2 * n + 1) + k;
                    let nmk1 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + n * (2 * n + 1) + k - 1;
                    let ank = (n + k) / (n - k + 1);//double
                    Dnmd[nmk1] = Math.sqrt(ank) * Math.tan(alpha * 0.5) * Dnmd[nmk];
                    

                }
                for (let m = n; m >= 1; m--) {
                    for (let k = m - 1; k >= 1 - m; k--) {
                        nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                        let nmk1 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k + 1;
                        let nm1k = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + (m - 1) * (2 * n + 1) + k;
                        Dnmd[nm1k] = anmk[1][nmk] * Dnmd[nmk1] + anmk[0][nmk] * sc * Dnmd[nmk];
                        //if(nm1k==582){throw "here"}
                        if (i == 27 && nm1k == 597) {
                            console.log(`alpha=${alpha} | sc=${sc} |nmk=${nmk}`)
                            console.log(`${anmk[1][nmk]} * ${Dnmd[nmk1]} + ${anmk[0][nmk]} * ${sc} * ${Dnmd[nmk]}`)
                        }
                    }
                }
            }


            for (let n = 1; n < core.numExpansions; n++) {
                for (let m = 0; m <= n; m++) {
                    for (let k = -m; k <= -1; k++) {
                        let ek = Math.pow(-1.0, k);
                        let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                        let nmk1 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) - k * (2 * n + 1) - m;
                        Dnmd[nmk] = ek * Dnmd[nmk];
                        Dnmd[nmk1] = Math.pow(-1.0, m + k) * Dnmd[nmk];

                    }
                    for (let k = 0; k <= m; k++) {
                        let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                        let nmk1 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + k * (2 * n + 1) + m;
                        let nmk2 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) - k * (2 * n + 1) - m;
                        Dnmd[nmk1] = Math.pow(-1.0, m + k) * Dnmd[nmk];
                        Dnmd[nmk2] = Dnmd[nmk1];
                        //console.log(`Dnmd[${nmk1}]: ${Dnmd[nmk1]}`);
                    }
                }
            }

            for (let n = 0; n < core.numExpansions; n++) {
                for (let m = 0; m <= n; m++) {
                    for (let k = -n; k <= n; k++) {
                        let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                        let nk = n * (n + 1) + k;
                        let c = Complex.fromBuffer(expBeta, k + m + 2 * core.numExpansions - 2);
                        this.Dnm[i][m][nk * 2] = Dnmd[nmk] * c.re;
                        this.Dnm[i][m][nk * 2 + 1] = Dnmd[nmk] * c.im;
                    }
                }
            }

            alpha = -alpha;
            beta = -beta;

            sc = Math.sin(alpha) / (1 + Math.cos(alpha));
            for (let n = 0; n < 4 * core.numExpansions - 3; n++) {
                let c = new Complex(0, (n - 2 * core.numExpansions + 2) * beta);
                let c2 = c.exp();
                expBeta[n * 2] = c2.re;
                expBeta[n * 2 + 1] = c2.im;
            }

            for (let n = 0; n < core.numExpansions; n++) {
                let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + n * (2 * n + 1) + n;
                Dnmd[nmk] = Math.pow(Math.cos(alpha * 0.5), 2 * n);
                for (let k = n; k >= 1 - n; k--) {
                    nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + n * (2 * n + 1) + k;
                    let nmk1 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + n * (2 * n + 1) + k - 1;
                    let ank = (n + k) / (n - k + 1);//double
                    Dnmd[nmk1] = Math.sqrt(ank) * Math.tan(alpha * 0.5) * Dnmd[nmk];

                }
                for (let m = n; m >= 1; m--) {
                    for (let k = m - 1; k >= 1 - m; k--) {
                        nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                        let nmk1 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k + 1;
                        let nm1k = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + (m - 1) * (2 * n + 1) + k;
                        Dnmd[nm1k] = anmk[1][nmk] * Dnmd[nmk1] + anmk[0][nmk] * sc * Dnmd[nmk];

                    }
                }
            }

            for (let n = 1; n < core.numExpansions; n++) {
                for (let m = 0; m <= n; m++) {
                    for (let k = -m; k <= -1; k++) {
                        let ek = Math.pow(-1.0, k);
                        let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                        let nmk1 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) - k * (2 * n + 1) - m;
                        Dnmd[nmk] = ek * Dnmd[nmk];
                        Dnmd[nmk1] = Math.pow(-1.0, m + k) * Dnmd[nmk];
                    }
                    for (let k = 0; k <= m; k++) {
                        let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                        let nmk1 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + k * (2 * n + 1) + m;
                        let nmk2 = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) - k * (2 * n + 1) - m;
                        Dnmd[nmk1] = Math.pow(-1.0, m + k) * Dnmd[nmk];
                        Dnmd[nmk2] = Dnmd[nmk1];
                    }
                }
            }

            for (let n = 0; n < core.numExpansions; n++) {
                for (let m = 0; m <= n; m++) {
                    for (let k = -n; k <= n; k++) {
                        let nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k;
                        let nk = n * (n + 1) + k;
                        let c = Complex.fromBuffer(expBeta, k + m + 2 * core.numExpansions - 2);
                        this.Dnm[i + numRelativeBox][m][nk * 2] = Dnmd[nmk] * c.re;
                        this.Dnm[i + numRelativeBox][m][nk * 2 + 1] = Dnmd[nmk] * c.im;
                    }
                }
            }
        }

        // for (j = 0; j < numBoxIndexTotal; j++) {
        //     for (i = 0; i < numCoefficients; i++) {
        //         Mnm[j][i] = 0;
        //     }
        // }


        //debug
        this.Dnmd = Dnmd;
        this.expBeta = expBeta;
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
    // Spherical harmonic rotation
    rotation(Cnm: Float32Array, Dnm: Array<Float32Array>): Float32Array {
        const CnmOut = new Float32Array(Cnm.length);

        for (let n = 0; n < this.core.numExpansions; n++) {
            for (let m = 0; m <= n; m++) {
                let nms = n * (n + 1) / 2 + m;
                let CnmScalarRe = 0, CnmScalarIm = 0;
                for (let k = -n; k <= -1; k++) {
                    let nk = n * (n + 1) + k;
                    let nks = n * (n + 1) / 2 - k;
                    let t = Complex.fromBuffer(Dnm[m], nk)
                        .multiply(Complex.fromBuffer(Cnm, nks).conj())
                    CnmScalarRe += t.re;
                    CnmScalarIm += t.im;
                }
                for (let k = 0; k <= n; k++) {
                    let nk = n * (n + 1) + k;
                    let nks = n * (n + 1) / 2 + k;
                    let t = Complex.fromBuffer(Dnm[m], nk)
                        .multiply(Complex.fromBuffer(Cnm, nks))
                    CnmScalarRe += t.re;
                    CnmScalarIm += t.im;
                }
                CnmOut[nms * 2] = CnmScalarRe;
                CnmOut[nms * 2 + 1] = CnmScalarIm;
            }
        }
        return CnmOut;
    }



    async p2m(numBoxIndex: number, particleOffset: any) {
        const core = this.core;
        const particleBuffer = this.particleBuffer;



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
                        let eim = new Complex(0, -m * beta).exp();
                        const w = particleBuffer[j * 4 + 3];

                        MnmVec[nms * 2] += w * YnmReal[nm] * eim.re;
                        MnmVec[nms * 2 + 1] += w * YnmReal[nm] * eim.im;

                    }
                }
            }
            this.Mnm[jj] = MnmVec;
        }


    }
    async m2m(numBoxIndex: number, numBoxIndexOld: number, numLevel: number) {
        const core = this.core;
        const anm = this.anm;
        const boxSize = core.rootBoxSize / (1 << numLevel);
        const MnmVectorA = new Float32Array(core.numCoefficients * 2);
        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + core.levelOffset[numLevel - 1];
            this.Mnm[ib] = new Float32Array(core.numCoefficients * 2);
        }
        for (let jj = 0; jj < numBoxIndexOld; jj++) {
            let jb = jj + core.levelOffset[numLevel];
            let nfjp = Math.trunc(core.boxIndexFull[jb] / 8);
            let nfjc = core.boxIndexFull[jb] % 8;
            let ib = core.boxIndexMask[nfjp] + core.levelOffset[numLevel - 1];
            let boxIndex3D = core.unmorton(nfjc);
            boxIndex3D.x = 4 - boxIndex3D.x * 2;
            boxIndex3D.y = 4 - boxIndex3D.y * 2;
            boxIndex3D.z = 4 - boxIndex3D.z * 2;
            let je = core.morton1(boxIndex3D, 3);
            let rho = boxSize * Math.sqrt(3.0) / 4;
            for (let j = 0; j < core.numCoefficients * 2; j++) {
                MnmVectorA[j] = this.Mnm[jb][j];
            }
            let MnmVectorB = this.rotation(MnmVectorA, this.Dnm[je]);
            for (let j = 0; j < core.numExpansions; j++) {
                for (let k = 0; k <= j; k++) {
                    let jk = j * j + j + k;
                    let jks = j * (j + 1) / 2 + k;
                    let MnmScalarRe = 0, MnmScalarIm = 0;
                    for (let n = 0; n <= j - k; n++) {
                        let jnk = (j - n) * (j - n) + j - n + k;
                        let jnks = (j - n) * (j - n + 1) / 2 + k;
                        let nm = n * n + n;
                        let temp = Math.pow(-1.0, n) * anm[nm] * anm[jnk] / anm[jk] * Math.pow(rho, n);
                        let cnm = Complex.fromBuffer(this.Ynm, nm).multiply(new Complex(temp, 0));
                        let temp2 = Complex.fromBuffer(MnmVectorB, jnks).multiply(cnm);

                        MnmScalarRe += temp2.re;
                        MnmScalarIm += temp2.im;
                    }
                    MnmVectorA[jks * 2] = MnmScalarRe;
                    MnmVectorA[jks * 2 + 1] = MnmScalarIm;
                }
            }

            MnmVectorB = this.rotation(MnmVectorA, this.Dnm[je + numRelativeBox]);
            for (let j = 0; j < core.numCoefficients; j++) {
                this.Mnm[ib][j * 2] += MnmVectorB[j * 2];
                this.Mnm[ib][j * 2 + 1] += MnmVectorB[j * 2 + 1];
            }
        }
    }
    async m2l(numBoxIndex: number, numLevel: number) {
        // int i, j, ii, ib, ix, iy, iz, ij, jj, jb, jx, jy, jz, je, k, jk, jks, n, nk, nks, jkn, jnk;
        // vec3<int> boxIndex3D;
        // vec3<double> dist;
        // double boxSize, rho, rhoj, rhojk, rhojn;
        // std::complex<double> LnmVectorA[numCoefficients], MnmVectorA[numCoefficients];
        // std::complex<double> LnmVectorB[numCoefficients], MnmVectorB[numCoefficients];
        // std::complex<double> cnm, LnmScalar;
        const core = this.core;










        let boxSize = core.rootBoxSize / (1 << numLevel);
        const LnmVectorA = new Float32Array(core.numCoefficients * 2);
        // if (numLevel == 2) {

        // }
        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + core.levelOffset[numLevel - 1];
            let boxIndex3D = core.unmorton(core.boxIndexFull[ib]);
            let ix = boxIndex3D.x,
                iy = boxIndex3D.y,
                iz = boxIndex3D.z;
            for (let ij = 0; ij < core.numInteraction[ii]; ij++) {
                let jj = core.interactionList[ii][ij];
                let jb = jj + core.levelOffset[numLevel - 1];

                const MnmVectorB = this.Mnm[jb];
                let boxIndex3D = core.unmorton(core.boxIndexFull[jb]);
                let jx = boxIndex3D.x,
                    jy = boxIndex3D.y,
                    jz = boxIndex3D.z;
                let dx = (ix - jx) * boxSize,
                    dy = (iy - jy) * boxSize,
                    dz = (iz - jz) * boxSize;
                boxIndex3D.x = (ix - jx) + 3;
                boxIndex3D.y = (iy - jy) + 3;
                boxIndex3D.z = (iz - jz) + 3;
                let je = core.morton1(boxIndex3D, 3);
                let rho = Math.sqrt(dx * dx + dy * dy + dz * dz) + eps;
                const MnmVectorA = this.rotation(MnmVectorB, this.Dnm[je]);
                let rhoj = 1;
                for (let j = 0; j < core.numExpansions; j++) {
                    let rhojk = rhoj;
                    rhoj *= rho;
                    for (let k = 0; k <= j; k++) {
                        let jk = j * j + j + k;
                        let jks = j * (j + 1) / 2 + k;
                        let LnmScalarRe = 0, LnmScalarIm = 0;
                        let rhojn = rhojk;
                        rhojk *= rho;
                        for (let n = Math.abs(k); n < core.numExpansions; n++) {
                            rhojn *= rho;
                            let nk = n * n + n + k;
                            let nks = n * (n + 1) / 2 + k;
                            let jkn = jk * core.numExpansion2 + nk;
                            let jnk = (j + n) * (j + n) + j + n;
                            let cnm = new Complex(this.Anm[jkn] / rhojn, 0).multiply(Complex.fromBuffer(this.Ynm, jnk));
                            let temp = Complex.fromBuffer(MnmVectorA, nks).multiply(cnm);
                            LnmScalarRe += temp.re;
                            LnmScalarIm += temp.im;
                        }
                        LnmVectorA[jks * 2] = LnmScalarRe;
                        LnmVectorA[jks * 2 + 1] = LnmScalarIm;
                    }
                }
                let LnmVectorB = this.rotation(LnmVectorA, this.Dnm[je + numRelativeBox]);
                //console.log(LnmVectorB);

                for (let j = 0; j < core.numCoefficients; j++) {
                    this.Lnm[ii][j * 2] += LnmVectorB[j * 2];
                    this.Lnm[ii][j * 2 + 1] += LnmVectorB[j * 2 + 1];
                }
            }
        }
        for (let jj = 0; jj < numBoxIndex; jj++) {
            let jb = jj + core.levelOffset[numLevel - 1];
            this.Mnm[jb].fill(0);
        }

    }
    async l2l(numBoxIndex: number, numLevel: number) {

        //           int numBoxIndexOld, ii, ib, i, nfip, nfic, je, j, k, jk, jks, n, jnk, nk, nks;
        //   vec3<int> boxIndex3D;
        //   double boxSize, rho;
        //   std::complex<double> cnm, LnmScalar;
        //   std::complex<double> LnmVectorA[numCoefficients], LnmVectorB[numCoefficients];
        const core = this.core;
        const boxSize = core.rootBoxSize / (1 << numLevel);

        let LnmVectorA = new Float32Array(core.numCoefficients * 2);
        let numBoxIndexOld = numBoxIndex;
        if (numBoxIndexOld < 8) { numBoxIndexOld = 8; }
        for (let ii = 0; ii < numBoxIndexOld; ii++) {
            this.LnmOld[ii] = this.Lnm[ii];
        }

        //int nbc, neo[numBoxIndexFull];
        let nbc = -1;
        let neo = new Int32Array(core.numBoxIndexFull);
        numBoxIndexOld = 0;
        for (let i = 0; i < core.numBoxIndexFull; i++)
            neo[i] = -1;
        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + core.levelOffset[numLevel - 1];
            if (nbc != Math.trunc(core.boxIndexFull[ib] / 8)) {
                nbc = Math.trunc(core.boxIndexFull[ib] / 8);
                neo[nbc] = numBoxIndexOld;
                numBoxIndexOld++;
            }
        }

        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + core.levelOffset[numLevel - 1];
            let nfip = Math.trunc(core.boxIndexFull[ib] / 8);
            let nfic = core.boxIndexFull[ib] % 8;
            let boxIndex3D = core.unmorton(nfic);
            boxIndex3D.x = boxIndex3D.x * 2 + 2;
            boxIndex3D.y = boxIndex3D.y * 2 + 2;
            boxIndex3D.z = boxIndex3D.z * 2 + 2;
            let je = core.morton1(boxIndex3D, 3);
            let rho = boxSize * Math.sqrt(3.0) / 2;
            ib = neo[nfip];
            for (let i = 0; i < core.numCoefficients; i++) {
                LnmVectorA[i * 2] = this.LnmOld[ib][i * 2];
                LnmVectorA[i * 2 + 1] = this.LnmOld[ib][i * 2 + 1];
            }
            let LnmVectorB = this.rotation(LnmVectorA, this.Dnm[je]);
            for (let j = 0; j < core.numExpansions; j++) {
                for (let k = 0; k <= j; k++) {
                    let jk = j * j + j + k;
                    let jks = j * (j + 1) / 2 + k;
                    let LnmScalarRe = 0, LnmScalarIm = 0;
                    for (let n = j; n < core.numExpansions; n++) {
                        let jnk = (n - j) * (n - j) + n - j;
                        let nk = n * n + n + k;
                        let nks = n * (n + 1) / 2 + k;
                        let cnm = Complex.fromBuffer(this.Ynm, jnk).multiply(new Complex(this.anm[jnk] * this.anm[jk] / this.anm[nk] * Math.pow(rho, n - j)));
                        let temp = cnm.multiply(Complex.fromBuffer(LnmVectorB, nks))
                        LnmScalarRe += temp.re;
                        LnmScalarIm += temp.im;
                    }
                    LnmVectorA[jks * 2] = LnmScalarRe;
                    LnmVectorA[jks * 2 + 1] = LnmScalarIm;
                }
            }
            LnmVectorB = this.rotation(LnmVectorA, this.Dnm[je + numRelativeBox]);
            this.Lnm[ii] = LnmVectorB;
        }

    }
    async l2p(numBoxIndex: number) {


        //         int ii, i, n, nm, nms, m;
        //   vec3<int> boxIndex3D;
        //   vec3<float> boxCenter;
        //   vec3<double> accel, dist;
        //   double boxSize, r, theta, phi, accelR, accelTheta, accelPhi;
        //   double xx, yy, s2, fact, pn, p, p1, p2, rn;
        //   double YnmReal[numExpansion2], YnmRealTheta[numExpansion2];
        //   std::complex<double> LnmVector[numCoefficients];
        //   std::complex<double> rr, rtheta, rphi, I(0.0, 1.0), eim;
        const core = this.core;

        let YnmReal = new Float32Array(core.numExpansion2);
        let YnmRealTheta = new Float32Array(core.numExpansion2);
        const boxSize = core.rootBoxSize / (1 << core.maxLevel);
        for (let ii = 0; ii < numBoxIndex; ii++) {
            let boxIndex3D = core.unmorton(core.boxIndexFull[ii]);
            let boxCenterX = core.boxMinX + (boxIndex3D.x + 0.5) * boxSize,
                boxCenterY = core.boxMinY + (boxIndex3D.y + 0.5) * boxSize,
                boxCenterZ = core.boxMinZ + (boxIndex3D.z + 0.5) * boxSize;

            let LnmVector = this.Lnm[ii];
            for (let i = core.particleOffset[0][ii]; i <= core.particleOffset[1][ii]; i++) {
                let dx = core.particleBuffer[i * 4] - boxCenterX,
                    dy = core.particleBuffer[i * 4 + 1] - boxCenterY,
                    dz = core.particleBuffer[i * 4 + 2] - boxCenterZ;
                const { rho, alpha, beta } = this.cart2sph(dx, dy, dz);
                let r = rho, theta = alpha, phi = beta;
                let xx = Math.cos(theta),
                    yy = Math.sin(theta),
                    s2 = Math.sqrt((1 - xx) * (1 + xx)),
                    fact = 1,
                    pn = 1;
                for (let m = 0; m < core.numExpansions; m++) {
                    let p = pn;
                    let nm = m * m + 2 * m;
                    YnmReal[nm] = this.factorial[nm] * p;
                    let p1 = p;
                    p = xx * (2 * m + 1) * p;
                    YnmRealTheta[nm] = this.factorial[nm] * (p - (m + 1) * xx * p1) / yy;
                    for (let n = m + 1; n < core.numExpansions; n++) {
                        nm = n * n + n + m;
                        YnmReal[nm] = this.factorial[nm] * p;
                        let p2 = p1;
                        p1 = p;
                        p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
                        YnmRealTheta[nm] = this.factorial[nm] * ((n - m + 1) * p - (n + 1) * xx * p1) / yy;
                    }
                    pn = -pn * fact * s2;
                    fact += 2;
                }
                let accelR = 0,
                    accelTheta = 0,
                    accelPhi = 0,
                    rn = 1;
                for (let n = 0; n < core.numExpansions; n++) {
                    let nm = n * n + n;
                    let nms = n * (n + 1) / 2;
                    let rr = new Complex(n * rn / r * YnmReal[nm], 0);
                    let rtheta = new Complex(rn * YnmRealTheta[nm], 0);
                    accelR += rr.multiply(Complex.fromBuffer(LnmVector, nms)).re;
                    accelTheta += rtheta.multiply(Complex.fromBuffer(LnmVector, nms)).re;
                    for (let m = 1; m <= n; m++) {
                        nm = n * n + n + m;
                        nms = n * (n + 1) / 2 + m;
                        let eim = new Complex(0, m * phi).exp();
                        rr = eim.multiplyReal(n * rn / r * YnmReal[nm]);
                        rtheta = eim.multiplyReal(rn * YnmRealTheta[nm]);
                        let rphi = eim.multiplyReal(m * rn * YnmReal[nm]).multiply(new Complex(0, 1));
                        let LnmVectorNms = Complex.fromBuffer(LnmVector, nms);
                        accelR += 2 * rr.multiply(LnmVectorNms).re;
                        accelTheta += 2 * rtheta.multiply(LnmVectorNms).re;
                        accelPhi += 2 * rphi.multiply(LnmVectorNms).re;
                    }
                    rn *= r;
                }
                const sin = Math.sin, cos = Math.cos;
                let accelX = sin(theta) * cos(phi) * accelR + cos(theta) * cos(phi) / r * accelTheta - sin(phi) / r / sin(theta) * accelPhi,
                    accelY = sin(theta) * sin(phi) * accelR + cos(theta) * sin(phi) / r * accelTheta + cos(phi) / r / sin(theta) * accelPhi,
                    accelZ = cos(theta) * accelR - sin(theta) / r * accelTheta;

                this.accelBuffer[i * 3] += inv4PI * accelX;
                this.accelBuffer[i * 3 + 1] += inv4PI * accelY;
                this.accelBuffer[i * 3 + 2] += inv4PI * accelZ;
            }
        }
    }

}