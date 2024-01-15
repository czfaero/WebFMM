import wgsl_p2p from '../shaders/FMM_p2p.wgsl';
import wgsl_p2m from '../shaders/FMM_p2m.wgsl';
import wgsl_m2m from '../shaders/FMM_m2m.wgsl';
import wgsl_m2l from '../shaders/FMM_m2l.wgsl';
import wgsl_l2l from '../shaders/FMM_l2l.wgsl';
import wgsl_l2p from '../shaders/FMM_l2p.wgsl';
import wgsl_buffer_sum from '../shaders/buffer_sum.wgsl';


import { IKernel } from './kernel';
import { FMMSolver } from '../FMMSolver';

import { Tester } from '../tester';

const eps = 1e-6;
const inv4PI = 0.25 / Math.PI;

const SIZEOF_32 = 4;

const numRelativeBox = 512;        // max of relative box positioning
const maxM2LInteraction = 189;


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

  static fromBuffer(b: Float64Array, i: number): Complex {
    return new Complex(b[i * 2], b[i * 2 + 1]);
  };
  constructor(re: number, im = 0) {
    this.re = re;
    this.im = im;
  }
}



export class KernelWgpu implements IKernel {
  core: FMMSolver;
  debug: boolean;
  particleCount: number;
  constructor(core: FMMSolver) {
    this.debug = false;
    this.core = core;
  }
  adapter: GPUAdapter;
  device: GPUDevice;
  particleBufferGPU: GPUBuffer;
  resultBufferGPU: GPUBuffer;
  commandBufferGPU: GPUBuffer;
  readBufferGPU: GPUBuffer;
  maxThreadCount: number;
  maxWorkgroupCount: number;
  accelBuffer: Float32Array;
  uniformBufferSize: number;
  uniformBufferGPU: GPUBuffer;
  particleOffsetGPU: GPUBuffer;
  factorialGPU: GPUBuffer;
  mnmBufferGPU: GPUBuffer;
  lnmBufferGPU: GPUBuffer;
  lnmOldBufferGPU: GPUBuffer;
  //ynmBufferGPU: GPUBuffer;
  dnmBufferGPU: GPUBuffer;
  mgBufferGPU: GPUBuffer;
  ngBufferGPU: GPUBuffer;

  debug_info: any;

  shaders: any;
  async Init(particleBuffer: Float32Array) {
    this.particleCount = particleBuffer.length / 4;
    this.accelBuffer = new Float32Array(this.particleCount * 3);
    this.adapter = await navigator.gpu.requestAdapter();
    this.device = await this.adapter.requestDevice();
    // to-do: check limit
    console.log(this.adapter);
    this.maxThreadCount = 256;
    this.maxWorkgroupCount = 256;
    // this.cmdBufferLength = this.maxThreadCount * this.maxWorkgroupCount * 2;// to-do: set a good value
    // this.cmdBufferSize = this.cmdBufferLength * SIZEOF_32;
    this.particleBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });


    this.uniformBufferSize = 16 * SIZEOF_32;// see shader

    this.uniformBufferGPU = this.device.createBuffer({
      size: this.uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.factorialGPU = this.device.createBuffer({
      size: this.core.numExpansions * 2 * SIZEOF_32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.mnmBufferGPU = this.device.createBuffer({
      size: this.core.numBoxIndexTotal * this.core.numCoefficients * 2 * SIZEOF_32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.lnmBufferGPU = this.device.createBuffer({
      size: this.core.numBoxIndexLeaf * this.core.numCoefficients * 2 * SIZEOF_32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.lnmOldBufferGPU = this.device.createBuffer({
      size: this.core.numBoxIndexLeaf * this.core.numCoefficients * 2 * SIZEOF_32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.resultBufferGPU = this.device.createBuffer({
      size: Math.max(this.maxThreadCount * this.maxWorkgroupCount * 3, this.particleCount * 3) * SIZEOF_32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.readBufferGPU = this.device.createBuffer({
      size: Math.max(this.resultBufferGPU.size, this.lnmBufferGPU.size),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    this.commandBufferGPU = this.device.createBuffer({
      size: 776192,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.device.queue.writeBuffer(this.particleBufferGPU, 0, particleBuffer);


    this.shaders = {
      p2p: wgsl_p2p,
      p2m: wgsl_p2m,
      m2m: wgsl_m2m,
      m2l: wgsl_m2l,
      l2l: wgsl_l2l,
      l2p: wgsl_l2p,
      sum: wgsl_buffer_sum,
    }
    const nameList = "p2p p2m m2m m2l l2l l2p sum".split(" ");
    for (const n of nameList) {
      this.shaders[n] = this.device.createShaderModule({
        code: this.shaders[n],
      })
    }

    if (this.debug) {
      this.debug_info = {};
      this.debug_info["particleBufferGPU"] = this.particleBufferGPU.size;
      this.debug_info["uniformBufferGPU"] = this.uniformBufferGPU.size;
    }

    await this.precalc();
  }

  async precalc() {
    const core = this.core;
    const Ynm = new Float64Array(4 * core.numExpansion2 * 2);
    /** complex [2 * numRelativeBox][numExpansions][numExpansion2]; for rotation -> m2m */
    const Dnm: Array<Array<Float64Array>> = new Array(2 * numRelativeBox);
    for (let i = 0; i < 2 * numRelativeBox; i++) {
      Dnm[i] = new Array(core.numExpansions);
      for (let j = 0; j < core.numExpansions; j++)
        Dnm[i][j] = new Float64Array(core.numExpansion2 * 2);
    }
    /** [2][numExpansion4]; for Dnmd -> Dnm */
    const anmk = [new Float64Array(core.numExpansion4), new Float64Array(core.numExpansion4)];
    /** [numExpansion4]; for Dnm */
    const Dnmd = new Float64Array(core.numExpansion4);
    /** complex [numExpansion2]; for Dnm */
    const expBeta = new Float64Array(core.numExpansion2 * 2);
    const factorial = new Float64Array(4 * core.numExpansion2);


    for (let n = 0; n < 2 * core.numExpansions; n++) {
      for (let m = -n; m <= n; m++) {
        let nm = n * n + n + m;
        const nabsm = Math.abs(m);
        let fnma = 1.0;
        for (let i = 1; i <= n - nabsm; i++)
          fnma *= i;
        let fnpa = 1.0;
        for (let i = 1; i <= n + nabsm; i++)
          fnpa *= i;
        factorial[nm] = Math.sqrt(fnma / fnpa);
      }
    }


    const Yn0 = new Float32Array(2 * core.numExpansions);
    let p0 = 1;
    let p1 = 1;
    Yn0[0] = p0;
    Yn0[1] = p1;
    for (let n = 1; n < 2 * core.numExpansions; n++) {
      let p = ((2 * n + 1) * p1 - n * p0) / (n + 1)
      Yn0[n + 1] = p;
      p0 = p1;
      p1 = p;
    }

    let pn = 1;
    for (let m = 0; m < 2 * core.numExpansions; m++) {
      let p = pn;
      let npn = m * m + 2 * m;
      let nmn = m * m;
      Ynm[npn * 2] = factorial[npn] * p;
      Ynm[nmn * 2] = Ynm[npn * 2];//conj(Ynm[npn*2])
      let p1 = p;
      p = (2 * m + 1) * p;
      for (let n = m + 1; n < 2 * core.numExpansions; n++) {
        let npm = n * n + n + m;
        let nmm = n * n + n - m;
        Ynm[npm * 2] = factorial[npm] * p;
        Ynm[nmm * 2] = Ynm[npm * 2];//conj(Ynm[npm*2]);
        let p2 = p1;
        p1 = p;
        p = ((2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
        //console.log(`${(2 * n + 1) * p1 - (n + m) * p2}/${n - m + 1}`)
      }
      pn = 0;
    }
    // console.log("Ynm")
    // console.log(Yn0)
    // console.log(Ynm)

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
      let xijc = boxIndex3D.x - 3;
      let yijc = boxIndex3D.y - 3;
      let zijc = boxIndex3D.z - 3;
      const rho = Math.sqrt(xijc * xijc + yijc * yijc + zijc * zijc) + eps;
      let alpha = Math.acos(zijc / rho);
      let beta;
      if (Math.abs(xijc) + Math.abs(yijc) < eps) {
        beta = 0;
      }
      else if (Math.abs(xijc) < eps) {
        beta = yijc / Math.abs(yijc) * Math.PI * 0.5;
      }
      else if (xijc > 0) {
        beta = Math.atan(yijc / xijc);
      }
      else {
        beta = Math.atan(yijc / xijc) + Math.PI;
      }

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
            Dnm[i][m][nk * 2] = Dnmd[nmk] * c.re;
            Dnm[i][m][nk * 2 + 1] = Dnmd[nmk] * c.im;
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
            Dnm[i + numRelativeBox][m][nk * 2] = Dnmd[nmk] * c.re;
            Dnm[i + numRelativeBox][m][nk * 2 + 1] = Dnmd[nmk] * c.im;
          }
        }
      }
    }

    // this.Ynm = new Float32Array(2 * core.numExpansion2);
    // for (let m = 0; m < core.numExpansions; m++) {
    //   for (let n = m; n < core.numExpansions; n++) {
    //     const npm = n * n + n + m;
    //     const nmm = n * n + n - m;
    //     this.Ynm[npm * 2 + 0] = Ynm[npm * 2];
    //     this.Ynm[nmm * 2 + 0] = Ynm[nmm * 2];
    //     this.Ynm[npm * 2 + 1] = Ynm[npm * 2 + 1];
    //     this.Ynm[nmm * 2 + 1] = Ynm[nmm * 2 + 1];
    //   }
    // }

    const DnmSize = (4 * core.numExpansion2 * core.numExpansions - core.numExpansions) / 3;
    this.Dnm = new Float32Array(2 * DnmSize * 2 * numRelativeBox);
    for (let je = 0; je < 2 * numRelativeBox; je++) {
      for (let n = 0; n < core.numExpansions; n++) {
        for (let m = 0; m <= n; m++) {
          for (let k = -n; k <= n; k++) {
            const nk = n * (n + 1) + k;
            const nmk = Math.trunc((4 * n * n * n + 6 * n * n + 5 * n) / 3) + m * (2 * n + 1) + k + je * DnmSize;
            this.Dnm[2 * nmk + 0] = Dnm[je][m][nk * 2];
            this.Dnm[2 * nmk + 1] = Dnm[je][m][nk * 2 + 1];
          }
        }
      }
    }
    //await Tester.VerifyFloatBuffer("data-hostDnm.bin", this.Dnm);
    //await Tester.VerifyFloatBuffer("data-hostYnm.bin", this.Ynm);


    //console.log(Ynm)

    // this.ynmBufferGPU = this.device.createBuffer({
    //   size: this.Ynm.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // });
    this.dnmBufferGPU = this.device.createBuffer({
      size: this.Dnm.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // this.device.queue.writeBuffer(this.ynmBufferGPU, 0, this.Ynm);
    this.device.queue.writeBuffer(this.dnmBufferGPU, 0, this.Dnm);

    // let threadsPerGroup = 64;
    // let ng = new Float32Array(threadsPerGroup);
    // let mg = new Float32Array(threadsPerGroup);
    // for (let n = 0; n < core.numExpansions; n++) {
    //   for (let m = 0; m <= n; m++) {
    //     let nms = n * (n + 1) / 2 + m;
    //     ng[nms] = n;
    //     mg[nms] = m;
    //   }
    // }
    // this.ngBufferGPU = this.device.createBuffer({
    //   size: ng.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // });
    // this.mgBufferGPU = this.device.createBuffer({
    //   size: mg.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // });
    // this.device.queue.writeBuffer(this.ngBufferGPU, 0, ng);
    // this.device.queue.writeBuffer(this.mgBufferGPU, 0, mg);
  }
  Dnm: Float32Array;
  // Ynm: Float32Array;
  async p2p(numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any) {
    if (this.debug) {
      this.debug_p2p_call_count = 0;
      this.debug_info["events"] = [];
      this.debug_info["events"].push({ time: performance.now(), tag: "start" });
    }

    const particleOffsetBuffer = new Uint32Array(numBoxIndex * 2);
    for (let i = 0; i < numBoxIndex; i++) {
      particleOffsetBuffer[i * 2] = particleOffset[0][i];
      particleOffsetBuffer[i * 2 + 1] = particleOffset[1][i];
    }
    this.particleOffsetGPU = this.device.createBuffer({
      size: particleOffsetBuffer.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(this.particleOffsetGPU, 0, particleOffsetBuffer);

    let commandCount = 0;
    const commandSize = 2;
    const maxCommandCount = this.maxThreadCount * this.maxWorkgroupCount;
    if (maxCommandCount > this.maxThreadCount * this.maxWorkgroupCount) {
      throw `Thread Count (${maxCommandCount}) > MaxThread(${this.maxThreadCount * this.maxWorkgroupCount})`;
    }
    const command = new Uint32Array(maxCommandCount * commandSize);




    if (this.debug) {
      this.debug_info["commandBufferGPU"] = this.commandBufferGPU.size;
      this.debug_info["particleOffsetGPU"] = this.particleOffsetGPU.size;
      this.debug_info["resultBufferGPU"] = this.resultBufferGPU.size;
      this.debug_info["readBufferGPU"] = this.readBufferGPU.size;
    }

    for (let ii = 0; ii < numBoxIndex; ii++) {
      for (let i = particleOffset[0][ii]; i <= particleOffset[1][ii]; i++) {
        for (let ij = 0; ij < numInteraction[ii]; ij++) {
          const jj = interactionList[ii][ij];
          command[commandCount * 2] = i;
          command[commandCount * 2 + 1] = jj;
          commandCount++;
          if (commandCount == maxCommandCount) {
            this.device.queue.writeBuffer(this.commandBufferGPU, 0, command);
            await this.p2p_ApplyAccel(command, commandCount);
            commandCount = 0;
            command.fill(0);
          }
        }
      }
    }

    this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0, commandCount * 2);
    await this.p2p_ApplyAccel(command, commandCount);
    if (this.debug) {
      this.debug_info["events"].push({ time: performance.now(), tag: "end" });
    }

    if (this.debug) {
      // await this.readBufferGPU.mapAsync(GPUMapMode.READ);
      // const handle = this.readBufferGPU.getMappedRange();
      // this.accelBuffer = new Float32Array(handle);
      // // this.readBufferGPU.unmap();
      console.log(`debug_p2p_call_count: ${this.debug_p2p_call_count}`);
      const startTime = this.debug_info["events"][0].time;
      this.debug_info["events"].forEach(o => o.time -= startTime);
      //console.log(JSON.stringify(this.debug_info))
      console.log(this.debug_info);
    }
  }

  debug_p2p_call_count: number;
  async p2p_ApplyAccel(cmdBuffer: Uint32Array, commandCount: number) {
    if (this.debug) {
      this.debug_info["events"].push({ time: performance.now(), tag: "prepare submit", i: this.debug_p2p_call_count, thread_count: commandCount });
      this.debug_p2p_call_count++;
    }
    const uniformBuffer = new Uint32Array(4);
    uniformBuffer.set([commandCount]);
    this.device.queue.writeBuffer(
      this.uniformBufferGPU,
      0,
      uniformBuffer
    );
    //console.log(cmdBuffer); throw "pause";
    await this.RunCompute("p2p",
      [this.uniformBufferGPU, this.particleBufferGPU, this.resultBufferGPU, this.commandBufferGPU, this.particleOffsetGPU],
      this.maxWorkgroupCount, this.resultBufferGPU, [this.resultBufferGPU]
    );
    //console.log("applyaccel");
    await this.readBufferGPU.mapAsync(GPUMapMode.READ);
    const handle = this.readBufferGPU.getMappedRange();
    let tempAccelBuffer = new Float32Array(handle);
    // console.log(cmdBuffer);
    // console.log(tempAccelBuffer);
    for (let i = 0; i < commandCount; i++) {
      let index = cmdBuffer[i * 2];
      this.accelBuffer[index * 3] += tempAccelBuffer[i * 3];
      this.accelBuffer[index * 3 + 1] += tempAccelBuffer[i * 3 + 1];
      this.accelBuffer[index * 3 + 2] += tempAccelBuffer[i * 3 + 2];
    }
    this.readBufferGPU.unmap();
  }

  /** only for debug. complex [numBoxIndexTotal][numCoefficients] */
  Mnm: Array<Float32Array>;
  /** only for debug. complex [numBoxIndexLeaf][numCoefficients] */
  Lnm: Array<Float32Array>;
  factorial: Float32Array;
  async p2m(numBoxIndex: number, particleOffset: any) {

    let fact = 1.0;
    let factorial = new Float32Array(2 * this.core.numExpansions);
    for (let m = 0; m < factorial.length; m++) {
      factorial[m] = fact;
      fact = fact * (m + 1);
    }
    this.device.queue.writeBuffer(this.factorialGPU, 0, factorial);
    this.factorial = factorial;


    // command (boxId)
    let command = new Uint32Array(numBoxIndex);
    for (let i = 0; i < numBoxIndex; i++) {
      command[i] = this.core.boxIndexFull[i];
    }
    this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0, numBoxIndex);

    const boxSize = this.core.rootBoxSize / (1 << this.core.maxLevel);

    let maxParticlePerBox = 0;
    for (let jj = 0; jj < numBoxIndex; jj++) {
      let c = particleOffset[1][jj] - particleOffset[0][jj] + 1;
      if (c > maxParticlePerBox) { maxParticlePerBox = c; }
    }

    const uniformBuffer = new Float32Array(4);
    uniformBuffer[0] = boxSize;
    uniformBuffer[1] = this.core.boxMinX;
    uniformBuffer[2] = this.core.boxMinY;
    uniformBuffer[3] = this.core.boxMinZ;
    const uniformBuffer2 = new Uint32Array(3);
    uniformBuffer2[0] = numBoxIndex;
    uniformBuffer2[1] = this.core.numExpansions;
    uniformBuffer2[2] = maxParticlePerBox;
    this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);
    this.device.queue.writeBuffer(this.uniformBufferGPU, uniformBuffer.byteLength, uniformBuffer2);

    await this.RunCompute("p2m",
      [this.uniformBufferGPU, this.particleBufferGPU, this.mnmBufferGPU, this.commandBufferGPU, this.particleOffsetGPU, this.factorialGPU],
      numBoxIndex, this.mnmBufferGPU
    );

    if (this.debug) {
      console.log("debug p2m");
      await this.readBufferGPU.mapAsync(GPUMapMode.READ);
      const handle = this.readBufferGPU.getMappedRange();
      let tempReadBuffer = new Float32Array(handle);
      this.Mnm = new Array(this.core.numBoxIndexTotal);
      for (let i = 0; i < numBoxIndex; i++) {
        const MnmVec = new Float32Array(this.core.numCoefficients * 2);
        for (let j = 0; j < this.core.numCoefficients * 2; j++) {
          MnmVec[j] = tempReadBuffer[i * this.core.numCoefficients * 2 + j];
        }
        this.Mnm[i] = MnmVec;
      }
      this.readBufferGPU.unmap();
      //console.log(this.Mnm)
    }

  }

  async m2m(numBoxIndex: number, numBoxIndexOld: number, numLevel: number) {
    const core = this.core;
    let command = new Int32Array(numBoxIndexOld * 3);
    const boxPerGroup = 1;
    const commandLength = 3;

    let command_sum = new Int32Array(1 << 3 * numLevel);

    for (let jj = 0; jj < numBoxIndexOld; jj++) {
      let jb = jj + core.levelOffset[numLevel];
      let nfjp = Math.trunc(core.boxIndexFull[jb] / 8);
      let nfjc = core.boxIndexFull[jb] % 8;
      let ib = core.boxIndexMask[nfjp] + core.levelOffset[numLevel - 1];// MnmIndex
      let boxIndex3D = core.unmorton(nfjc);
      boxIndex3D.x = 4 - boxIndex3D.x * 2;
      boxIndex3D.y = 4 - boxIndex3D.y * 2;
      boxIndex3D.z = 4 - boxIndex3D.z * 2;
      let je = core.morton1(boxIndex3D, 3);
      command[jj * commandLength + 0] = jb;//Mnm index
      command[jj * commandLength + 1] = je + 1;
      command[jj * commandLength + 2] = core.boxIndexFull[jb];//to-do: check for empty box or more level 
      //command[jj * commandLength + 3] = ib;
      //console.log(command[jj * 3 + 2])
      command_sum[nfjp] = ib;
    }
    //console.log(command)
    //console.log(command_sum)

    this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0, numBoxIndexOld * 3);
    const boxSize = this.core.rootBoxSize / (1 << numLevel);

    const uniformBuffer = new Float32Array(1);
    uniformBuffer[0] = boxSize;
    this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);

    if (this.resultBufferGPU.size < (1 << 3 * numLevel) * 8 * this.core.numCoefficients * 2 * SIZEOF_32) {
      throw "resultBufferGPU < numBoxIndexOld * numCoefficients * 2";
    }
    this.RunCompute("m2m",
      [this.uniformBufferGPU, this.mnmBufferGPU, this.commandBufferGPU
        , this.dnmBufferGPU, this.resultBufferGPU],
      numBoxIndexOld / boxPerGroup, this.resultBufferGPU, [this.resultBufferGPU]
    );

    if (this.debug) {


    }
    const numThread = numBoxIndexOld / 8 * core.numCoefficients * 2;
    const uniformBuffer_sum = new Int32Array(5);
    uniformBuffer_sum[0] = 8; //srcPackSize 
    uniformBuffer_sum[1] = numThread;
    uniformBuffer_sum[2] = 1;//commandLength
    uniformBuffer_sum[3] = 0;//commandTarget
    uniformBuffer_sum[4] = core.numCoefficients * 2;//vectorLength 
    this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer_sum);
    const numGroup = Math.ceil(numThread / 256);
    this.device.queue.writeBuffer(this.commandBufferGPU, 0, command_sum);
    await this.RunCompute("sum",
      [this.uniformBufferGPU, this.resultBufferGPU, this.mnmBufferGPU, this.commandBufferGPU],
      numGroup,
      this.mnmBufferGPU
    );

    if (this.debug) {
      console.log("debug m2m")
      await this.readBufferGPU.mapAsync(GPUMapMode.READ);
      const handle = this.readBufferGPU.getMappedRange();
      let tempReadBuffer = new Float32Array(handle);
      this.Mnm = new Array(this.core.numBoxIndexTotal);
      for (let i = 0; i < this.core.numBoxIndexTotal; i++) {
        const MnmVec = new Float32Array(this.core.numCoefficients * 2);
        for (let j = 0; j < this.core.numCoefficients * 2; j++) {
          MnmVec[j] = tempReadBuffer[i * this.core.numCoefficients * 2 + j];
        }
        this.Mnm[i] = MnmVec;
      }
      this.readBufferGPU.unmap();
      console.log("Mnm:")
      //console.log(this.Mnm);
      //throw "pause after m2m";
    }
  }
  async m2l(numBoxIndex: number, numLevel: number) {

    const core = this.core;
    const commandLength = 2 * maxM2LInteraction + 1;//count, pairs
    let command = new Int32Array(commandLength * numBoxIndex);

    for (let ii = 0; ii < numBoxIndex; ii++) {
      let ib = ii + core.levelOffset[numLevel - 1];
      let indexi = core.unmorton(core.boxIndexFull[ib]);
      let ix = indexi.x,
        iy = indexi.y,
        iz = indexi.z;

      command[ii * commandLength] = core.numInteraction[ii];
      for (let ij = 0; ij < core.numInteraction[ii]; ij++) {
        let jj = core.interactionList[ii][ij];
        let jbd = jj + core.levelOffset[numLevel - 1];
        let indexj = core.unmorton(core.boxIndexFull[jbd]);
        let jx = indexj.x, jy = indexj.y, jz = indexj.z;

        let je = core.morton1({ x: ix - jx + 3, y: iy - jy + 3, z: iz - jz + 3 }, 3);
        let jb = jj + core.levelOffset[numLevel - 1];
        command[ii * commandLength + 1 + ij * 2] = jb;//Mnm index
        command[ii * commandLength + 1 + ij * 2 + 1] = je + 1;

      }
    }
    // console.log("m2l command:");
    // console.log(command);
    this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0);

    const boxSize = this.core.rootBoxSize / (1 << numLevel);

    const uniformBuffer = new Float32Array(1);
    uniformBuffer[0] = boxSize;
    this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);
    this.RunCompute("m2l",
      [this.uniformBufferGPU, this.mnmBufferGPU, this.commandBufferGPU
        , this.dnmBufferGPU, this.lnmBufferGPU],
      numBoxIndex, this.lnmBufferGPU
    );
    if (this.debug) {
      console.log("debug m2l")
      {
        await this.readBufferGPU.mapAsync(GPUMapMode.READ);
        const handle = this.readBufferGPU.getMappedRange();
        let tempReadBuffer = new Float32Array(handle);
        {
          this.Lnm = new Array(this.core.numBoxIndexLeaf);
          for (let i = 0; i < this.core.numBoxIndexTotal; i++) {
            const LnmVec = new Float32Array(this.core.numCoefficients * 2);
            for (let j = 0; j < this.core.numCoefficients * 2; j++) {
              LnmVec[j] = tempReadBuffer[i * this.core.numCoefficients * 2 + j];
            }
            this.Lnm[i] = LnmVec;
          }
        }
        console.log("Lnm:")
        console.log(this.Lnm);
        this.readBufferGPU.unmap();
      }
      //throw "pause after m2l";
    }



  }
  async l2l(numBoxIndex: number, numLevel: number) {
    //console.log("l2l")
    const core = this.core;
    const commandLength = 2 ;
    let command = new Int32Array(commandLength * numBoxIndex);

    let nbc = -1, neo = new Array(core.numBoxIndexFull);
    let numBoxIndexOld = 0;

    for (let i = 0; i < core.numBoxIndexFull; i++) { neo[i] = -1; }
    for (let ii = 0; ii < numBoxIndex; ii++) {
      let ib = ii + core.levelOffset[numLevel - 1];
      if (nbc != Math.floor(core.boxIndexFull[ib] / 8)) {
        nbc = Math.floor(core.boxIndexFull[ib] / 8);
        neo[nbc] = numBoxIndexOld;
        numBoxIndexOld++;
      }
    }
    //console.log(neo);


    numBoxIndexOld = numBoxIndex;
    if (numBoxIndexOld < 8) { numBoxIndexOld = 8; }
    // for (let ii = 0; ii < numBoxIndexOld; ii++) {
    //   for (let i = 0; i < core.numCoefficients; i++) {
    //     //LnmOld[ii][i] = Lnm[ii][i];
    //   }
    // }

    for (let ii = 0; ii < numBoxIndex; ii++) {
      let ib = ii + core.levelOffset[numLevel - 1];
      let nfip = Math.floor(core.boxIndexFull[ib] / 8);
      let nfic = core.boxIndexFull[ib] % 8;
      let boxIndex3D = core.unmorton(nfic);
      boxIndex3D.x = boxIndex3D.x * 2 + 2;
      boxIndex3D.y = boxIndex3D.y * 2 + 2;
      boxIndex3D.z = boxIndex3D.z * 2 + 2;
      let je = core.morton1(boxIndex3D, 3);
      ib = neo[nfip];//source
      //console.log(`${ib}=>${ii}`)
      command[ii * commandLength] = ib;
      command[ii * commandLength + 1] = je + 1;
    }
    this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0);
    const boxSize = core.rootBoxSize / (1 << numLevel);
    const uniformBuffer = new Float32Array(1);
    uniformBuffer[0] = boxSize;
    this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);

    const preFunc = (commandEncoder: GPUCommandEncoder) => {
      commandEncoder.copyBufferToBuffer(this.lnmBufferGPU, 0, this.lnmOldBufferGPU, 0, this.lnmBufferGPU.size);
    }

    this.RunCompute("l2l",
      [this.uniformBufferGPU, this.lnmBufferGPU, this.commandBufferGPU,
      this.dnmBufferGPU, this.lnmOldBufferGPU],
      numBoxIndex, this.lnmBufferGPU, [], preFunc
    );

    if (this.debug) {
      console.log("debug l2l")
      {
        await this.readBufferGPU.mapAsync(GPUMapMode.READ);
        const handle = this.readBufferGPU.getMappedRange();
        let tempReadBuffer = new Float32Array(handle);
        {
          this.Lnm = new Array(this.core.numBoxIndexLeaf);
          for (let i = 0; i < this.core.numBoxIndexTotal; i++) {
            const LnmVec = new Float32Array(this.core.numCoefficients * 2);
            for (let j = 0; j < this.core.numCoefficients * 2; j++) {
              LnmVec[j] = tempReadBuffer[i * this.core.numCoefficients * 2 + j];
            }
            this.Lnm[i] = LnmVec;
          }
        }
        console.log("Lnm:")
        console.log(this.Lnm);
        this.readBufferGPU.unmap();
      }
      //throw "pause after l2l";
    }

  }
  async l2p(numBoxIndex: number) {
    const core = this.core;
    const commandLength = 4;//
    let command = new Int32Array(commandLength * numBoxIndex);
    let boxSize = core.rootBoxSize / (1 << core.maxLevel);
    const threadsPerGroup = 256;
    // loop foreach box set group
    let groupCount = 0;
    for (let ii = 0; ii < numBoxIndex; ii++) {
      let nParticle = core.particleOffset[1][ii] - core.particleOffset[0][ii];
      let nGroup = (nParticle + threadsPerGroup) / threadsPerGroup;
      nGroup = Math.floor(nGroup);
      for (let n = 0; n < nGroup; n++) {
        command[groupCount * commandLength + 0] = ii;
        command[groupCount * commandLength + 1] = core.particleOffset[0][ii] + n * threadsPerGroup;
        command[groupCount * commandLength + 2] = (n == nGroup - 1) ? threadsPerGroup : (nParticle - (nGroup - 1) * threadsPerGroup);
        command[groupCount * commandLength + 2] = core.boxIndexFull[ii];
        groupCount++;
      }

    }
    this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0);

    await this.RunCompute("l2p",
      [this.uniformBufferGPU, this.particleBufferGPU, this.resultBufferGPU, this.commandBufferGPU,
      this.factorialGPU, this.lnmBufferGPU,],
      groupCount, this.resultBufferGPU, [this.resultBufferGPU]
    );
    await this.readBufferGPU.mapAsync(GPUMapMode.READ);
    const handle = this.readBufferGPU.getMappedRange();
    let tempAccelBuffer = new Float32Array(handle);
    // console.log(cmdBuffer);
    console.log(tempAccelBuffer);
    console.log(this.accelBuffer);
    for (let i = 0; i < this.particleCount; i++) {

      this.accelBuffer[i * 3] += tempAccelBuffer[i * 3];
      this.accelBuffer[i * 3 + 1] += tempAccelBuffer[i * 3 + 1];
      this.accelBuffer[i * 3 + 2] += tempAccelBuffer[i * 3 + 2];
    }
    this.readBufferGPU.unmap();

  }

  async RunCompute(entryPoint: string, buffers: Array<GPUBuffer>, workgroupCount = 1, readBuffer: GPUBuffer = null, buffersToClear: Array<GPUBuffer> = null, preFunc: any = null) {
    const shaderModule = this.shaders[entryPoint];
    const computePipeline = this.device.createComputePipeline({
      layout: 'auto', // infer from shader code.
      compute: {
        module: shaderModule,
        entryPoint: entryPoint
      }
    });

    const bindGroupLayout = computePipeline.getBindGroupLayout(0);

    const entries = buffers.map((b, i) => {
      return { binding: i, resource: { buffer: b } };
    });
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: entries
    });
    const commandEncoder = this.device.createCommandEncoder();
    if (preFunc) {
      preFunc(commandEncoder);
    }
    if (buffersToClear) {
      for (const b of buffersToClear) {
        commandEncoder.clearBuffer(b);
      }
    }

    const computePassEncoder = commandEncoder.beginComputePass();
    computePassEncoder.setPipeline(computePipeline);
    computePassEncoder.setBindGroup(0, bindGroup);
    computePassEncoder.dispatchWorkgroups(workgroupCount);
    computePassEncoder.end();

    if (readBuffer) {
      commandEncoder.copyBufferToBuffer(readBuffer, 0, this.readBufferGPU, 0, readBuffer.size);
    }
    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);
    if (!readBuffer) {
      await this.device.queue.onSubmittedWorkDone();
    }
  }
}


function CompareNumber(a: number, b: number, delta = 0.002) {
  return Math.abs(a - b) < delta
}