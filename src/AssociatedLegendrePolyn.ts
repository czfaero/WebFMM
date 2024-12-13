// 
import { cart2sph } from "./utils";

/**
 * Calc Associated Legendre polynomials for given x=cos(theta) and p.  
 * @returns {Float32Array} Pnm: P00,P10,P11...  
 * 0<=n<p, 0<=m<=n  
 * Size: p*(p+1)/2
 */
export function CalcALP(p: number, x: number): Float32Array {
    const sqrt = Math.sqrt;
    let i: number;
    const max_n = p - 1;


    const size = p * (p + 1) / 2;
    const buffer = new Float32Array(size);
    const sinTheta = sqrt(1 - x * x);

    let Pnn = 1; // start from P00
    buffer[0] = Pnn;
    for (let m = 0; m < max_n;) { // 最后的Pnn n=max_n被上一轮计算
        let n = m;

        let P_pre2 = Pnn;

        n++;
        let P_pre1 = x * (2 * m + 1) * Pnn;// Recurrence formula (2)
        i = n * (n + 1) / 2 + m; buffer[i] = P_pre1;

        for (; n < max_n;) {
            const P_current = ((2 * n + 1) * x * P_pre1 - (n + m) * P_pre2) / (n - m + 1);// Recurrence formula (3)
            n++;
            i = n * (n + 1) / 2 + m; buffer[i] = P_current;

            // update preview P
            P_pre2 = P_pre1;
            P_pre1 = P_current;
        }
        Pnn = -(2 * m + 1) * sinTheta * Pnn;// Recurrence formula (1)
        m++; n = m;// should = loop + 1
        i = n * (n + 1) / 2 + m; buffer[i] = Pnn;
    }
    return buffer;
}

/**
 * Calcuate something about X=(r,theta,phi)   
 * As a piece of data is calc, "func" will be called with:  
 * n:  0 <= n < numExpansions  
 * m: -n <= m <= n  
 * m_abs: abs(m)  
 * r_n : r^n  
 * p : Associated Legendre polynomials for n, m at x=cos(theta).  
 * p_d: Derivative of Associated Legendre polynomials. By theta at x. 
 * @param numExpansions
 * @param theta Spherical coordinate, 0<= theta <= PI
 * @param r Spherical coordinate
 * @param {function} func callback function
 */
export function CalcALP_R(numExpansions: number, theta: number, r: number, func: Function) {
    const sqrt = Math.sqrt;
    const abs = Math.abs
    const eps = 1e-6;
    let i: number;
    const max_n = numExpansions - 1;

    const x = Math.cos(theta);
    let sinTheta = Math.sin(theta);


    if (sinTheta < eps) {
        sinTheta = eps;
    }
    if (sinTheta < 0) {
        throw "sinTheta<0"
    }



    let Pnn = 1; // start from P00
    let r_m = 1; // r^m
    let m = 0;
    let P_pre2;
    let P_pre1;

    // for m=0, call func only once
    {
        let n = 0, r_n = r_m;
        n++; r_n = r_n * r;

        let Pnn_next = x * (2 * m + 1) * Pnn; // Recurrence formula (2)

        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (sinTheta) // Recurrence formula (4)
        let Pnn_next_deriv = (n * x * Pnn_next - (n + m) * Pnn) / sinTheta // Recurrence formula (5)

        func(0, m, m, r_m, Pnn, Pnn_deriv);
        func(n, m, m, r_n, Pnn_next, Pnn_next_deriv);

        P_pre2 = Pnn;
        P_pre1 = Pnn_next;
        for (; n < max_n;) {
            n++; r_n = r_n * r;
            const P_current = ((2 * n - 1) * x * P_pre1 - (n + m - 1) * P_pre2) / (n - m);// Recurrence formula (3)
            const P_deriv = (n * x * P_current - (n + m) * P_pre1) / (sinTheta) // Recurrence formula (5)
            func(n, m, m, r_n, P_current, P_deriv);
            P_pre2 = P_pre1;
            P_pre1 = P_current;
        }

    }
    // start from m++ -> 1
    for (; m < max_n - 1;) {
        Pnn = -(2 * m + 1) * sinTheta * Pnn;// Recurrence formula (1)
        m++; r_m = r_m * r;
        let n = m, r_n = r_m;
        let Pnn_next = x * (2 * m + 1) * Pnn; // Recurrence formula (2)
        n++;
        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (sinTheta) // Recurrence formula (4)
        let Pnn_next_deriv = (n * x * Pnn_next - (n + m) * Pnn) / (sinTheta) // Recurrence formula (5)
        func(n - 1, m, m, r_n, Pnn, Pnn_deriv);
        func(n - 1, -m, m, r_n, Pnn, Pnn_deriv);
        r_n = r_n * r;
        func(n, m, m, r_n, Pnn_next, Pnn_next_deriv);
        func(n, -m, m, r_n, Pnn_next, Pnn_next_deriv);

        P_pre2 = Pnn;
        P_pre1 = Pnn_next;
        for (; n < max_n;) {
            n++; r_n = r_n * r;
            const P_current = ((2 * n - 1) * x * P_pre1 - (n + m - 1) * P_pre2) / (n - m);// Recurrence formula (3)
            const P_deriv = (n * x * P_current - (n + m) * P_pre1) / (sinTheta) // Recurrence formula (5)
            func(n, m, m, r_n, P_current, P_deriv);
            func(n, -m, m, r_n, P_current, P_deriv);
            P_pre2 = P_pre1;
            P_pre1 = P_current;
        }
    }
    // m=n
    {
        Pnn = -(2 * m + 1) * sinTheta * Pnn;// Recurrence formula (1)
        m++; r_m = r_m * r;
        let n = m + 1;// r_n = r_m * r;
        let Pnn_next = x * (2 * m + 1) * Pnn; // Recurrence formula (2)
        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (sinTheta) // Recurrence formula (4)
        func(n - 1, m, m, r_m, Pnn, Pnn_deriv);
        func(n - 1, -m, m, r_m, Pnn, Pnn_deriv);
    }
}

/**
 * Test functions above using "Reparameterization in terms of angles", only for sin(theta)>=0
 *  */
export function CalcALP_Test(theta) {
    console.log(`Test CalcALP for theta=${theta}`)
    const sin = Math.sin, cos = Math.cos;
    let s = sin(theta), c = cos(theta);

    const numExpansions = 5;
    const bufferSize = numExpansions * (numExpansions + 1) / 2;
    const testCase = new Float32Array(bufferSize);
    const testCase_derivative = new Float32Array(bufferSize);
    const i2n = new Array(bufferSize), i2m = new Array(bufferSize);

    [
        [0, 0, 1],
        [1, 0, c],
        [1, 1, -s],
        [2, 0, 1.5 * c * c - 0.5],
        [2, 1, -3 * c * s],
        [2, 2, 3 * s * s],
        [3, 0, 2.5 * c * c * c - 1.5 * c],
        [3, 1, -1.5 * (5 * c * c - 1) * s],
        [3, 2, 15 * c * s * s],
        [3, 3, -15 * s * s * s],
        [4, 0, (35 * c * c * c * c - 30 * c * c + 3) / 8],
        [4, 1, -2.5 * (7 * c * c * c - 3 * c) * s],
        [4, 2, 7.5 * (7 * c * c - 1) * s * s],
        [4, 3, -105 * c * s * s * s],
        [4, 4, 105 * s * s * s * s],
    ].forEach(data => {
        let n = data[0], m = data[1], P = data[2];
        let i = n * (n + 1) / 2 + m;
        i2n[i] = n;
        i2m[i] = m;
        testCase[i] = P;
    });

    [
        [0, 0, 0],
        [1, 0, -s],
        [1, 1, -c],
        [2, 0, -3 * c * s],
        [2, 1, 3 * (s * s - c * c)],
        [2, 2, 6 * s * c],
        [3, 0, 1.5 * s - 7.5 * c * c * s],
        [3, 1, 15 * c * s * s - 7.5 * c * c * c + 1.5 * c],
        [3, 2, 30 * c * c * s - 15 * s * s * s],
        [3, 3, -45 * s * s * c],
        [4, 0, 7.5 * c * s - 17.5 * c * c * c * s],
        [4, 1, -17.5 * c * c * c * c + 7.5 * c * c + 52.5 * c * c * s * s - 7.5 * s * s],
        [4, 2, -105 * c * s * s * s + 105 * s * c * c * c - 15 * s * c],
        [4, 3, 105 * s * s * s * s - 315 * c * c * s * s],
        [4, 4, 420 * s * s * s * c],

    ].forEach(data => {
        let n = data[0], m = data[1], P = data[2];
        let i = n * (n + 1) / 2 + m;
        testCase_derivative[i] = P;
    });


    // const getP = (n, m) => testCase[n * (n + 1) / 2 + m];

    const test1 = CalcALP(numExpansions, c);
    const test2 = new Float32Array(bufferSize);
    const test2_derivative = new Float32Array(bufferSize);
    const test_R = 1.1;
    let counter = 0;
    CalcALP_R(numExpansions, theta, test_R, (n, m, m_abs, r_n, P, P_deriv) => {
        let i = n * (n + 1) / 2 + m_abs;
        test2[i] = P;
        test2_derivative[i] = P_deriv;
        counter++;

        if (!CompareNumber(r_n, Math.pow(test_R, n))) {
            debugger;
        }
        // if (i == 4) {
        //     let p_d = (2 * c * getP(2, 1) - 3 * getP(1, 1)) / (-s * s) * (-s);
        //     debugger;
        // }
    });
    if (counter != numExpansions * numExpansions) { debugger; }

    const errors1 = VerifyFloatBuffer(testCase, test1);
    const errors2 = VerifyFloatBuffer(testCase, test2);
    const errors3 = VerifyFloatBuffer(testCase_derivative, test2_derivative);
    [errors1, errors2, errors3].forEach(errors => {
        errors.forEach(record => {
            record.n = i2n[record.i];
            record.m = i2m[record.i];
        });
    })
    if (errors1.length != 0) {
        console.log("Test for CalcALP():", errors1);
        console.log(Array.from(test1).map((v, i) => { return { n: i2n[i], m: i2m[i], v: v, i: i } }));
    }
    if (errors2.length != 0) {
        console.log("Test for CalcALP_R():", errors2);
        console.log(Array.from(test2).map((v, i) => { return { n: i2n[i], m: i2m[i], v: v, i: i } }));
    }

    if (errors3.length != 0) {
        console.log("Test for CalcALP_R() derivative:", errors3);
        console.log(Array.from(test2_derivative).map((v, i) => { return { n: i2n[i], m: i2m[i], v: v, i: i } }));
        debugger;
    }
}
function CompareNumber(a: number, b: number, delta = 0.002) {
    return Math.abs(a - b) < delta
}
function VerifyFloatBuffer(expect: ArrayLike<number>, data: ArrayLike<number>, max_error = 0.001) {

    if (data.length != expect.length) {
        console.log(data);
        throw `size: ${data.length}!=${expect.length}`;
    }
    let error_count = 0;
    let errors = [];
    for (let i = 0; i < data.length; i++) {
        const r = CompareNumber(expect[i], data[i], max_error);
        if (!r) {
            error_count++;
            //console.log(`[${i}]Expect: ${expect[i]} | Got: ${data[i]} |${expect[i] - data[i]}`);
            errors.push({ i: i, expect: expect[i], got: data[i], error: expect[i] - data[i] });
            if (error_count > 100) {
                break;
            }
        }
    }
    if (error_count == 0) {
        console.log("Success ");
    }
    else {
        console.log("Failure");
    }
    return errors;
}

/**
 * Verify $frac{1}{r^\prime}=\sum_{n=0}^\infty \frac{\rho^n}{r^{n+1}}P_n(u)$
 * @param {object} Q {x,y,z} src (here is the charge)
 * @param {object} P {x,y,z} dst
 */
export function Test_MultipoleExpansion(Q, P) {
    console.log("Test_MultipoleExpansion:", Q, P);
    const sin = Math.sin, cos = Math.cos;
    const _q = cart2sph(Q), _p = cart2sph(P);
    const rho = _q.x, alpha = _q.y, beta = _q.z;
    const r = _p.x, theta = _p.y, phi = _p.z;
    console.log(`Spherical:
Q(${rho},${alpha},${beta})
P(${r},${theta},${phi})`);

    const numExpansions = 10;

    const getDist = (a, b) => { return { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z }; }
    const getLength = (vec) => Math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    const r_dash = getLength(getDist(P, Q));
    const left = 1 / r_dash;


    const cosGamma = cos(theta) * cos(alpha) + sin(theta) * sin(alpha) * cos(phi - beta);

    const Pnm = CalcALP(numExpansions, cosGamma);
    let result = 0;
    let mu = rho / r;
    const m = 0;
    for (let n = 0; n < numExpansions; n++) {
        let i = n * (n + 1) / 2 + m;
        result += Pnm[i] * Math.pow(mu, n) / r;
    }
    console.log(`Left: `, left, "  r'=", r_dash)
    console.log(`Right: `, result)
    console.log(`error: `, left - result);
    console.log(`Test_MultipoleExpansion End`);
    // debugger;
}

/**
 * Verify P(\cos\gamma)=\sum_{m=-n}^n  Y_{n}^{-m}(\alpha,\beta) \cdot Y_{n}^{m}(\theta,\phi)
 * @param {object} Q {x,y,z} src
 * @param {object} P {x,y,z} dst
 */
export function Test_AdditionTheorem(Q, P) {
    console.log("Test_AdditionTheorem:", Q, P);
    const sin = Math.sin, cos = Math.cos;
    const _q = cart2sph(Q), _p = cart2sph(P);
    const rho = _q.x, alpha = _q.y, beta = _q.z;
    const r = _p.x, theta = _p.y, phi = _p.z;


    const cosGamma = cos(theta) * cos(alpha) + sin(theta) * sin(alpha) * cos(phi - beta);

    const numExpansions = 10;
    const Pnm = CalcALP(numExpansions, cosGamma);

    const Pnm_alpha = CalcALP(numExpansions, cos(alpha));
    const Pnm_theta = CalcALP(numExpansions, cos(theta));
    let factorial = new Float32Array(2 * numExpansions);
    for (let m = 0, fact = 1.0; m < factorial.length; m++) {
        factorial[m] = fact;
        fact = fact * (m + 1);
    }

    for (let n = 0; n < numExpansions; n++) {
        let i = n * (n + 1) / 2 + 0;
        const P_n = Pnm[i];
        let real = 0, imag = 0;
        for (let m = -n; m <= n; m++) {
            let abs_m = Math.abs(m);
            i = n * (n + 1) / 2 + abs_m;
            let fact = factorial[n - abs_m] / factorial[n + abs_m];
            let same = fact * Pnm_alpha[i] * Pnm_theta[i];
            let re = cos(-m * beta) * cos(m * phi) - sin(-m * beta) * sin(m * phi);
            let im = cos(-m * beta) * sin(m * phi) + sin(-m * beta) * cos(m * phi);
            real += re * same;
            imag += im * same;
        }
        const error = P_n - real;
        console.log(`P_${n}=`, P_n, " error=", error, "\nreal=", real, "imag", imag);
    }
    console.log(`Test_AdditionTheorem End`);
}