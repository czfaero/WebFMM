// 


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
     * Call the func, with:  
     * n:  0 <= n < numExpansions  
     * m: -n <= m <= n  
     * m_abs: abs(m)
     * r_n : r^n  
     * p : Associated Legendre polynomials for n, m at x.  
     * p_d: Derivative of p at x. 
     * @param numExpansions
     * @param x cos(theta)
     * @param func 
     */
export function CalcALP_R(numExpansions: number, x: number, r: number, func: Function) {
    const sqrt = Math.sqrt;
    let i: number;
    const max_n = numExpansions - 1;

    const x2 = x * x;
    const sinTheta = sqrt(1 - x2);

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

        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (1 - x2) // Recurrence formula (4)
        let Pnn_next_deriv = (n * x * Pnn_next - (n + m) * Pnn) / (x2 - 1) // Recurrence formula (5)

        func(0, m, m, r_m, Pnn, Pnn_deriv);
        func(n, m, m, r_n, Pnn_next, Pnn_next_deriv);

        P_pre2 = Pnn;
        P_pre1 = Pnn_next;
        for (; n < max_n;) {
            n++; r_n = r_n * r;
            const P_current = ((2 * n - 1) * x * P_pre1 - (n + m - 1) * P_pre2) / (n - m);// Recurrence formula (3)
            const P_deriv = (n * x * P_current - (n + m) * P_pre1) / (x2 - 1) // Recurrence formula (5)
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

        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (1 - x2) // Recurrence formula (4)
        let Pnn_next_deriv = (n * x * Pnn_next - (n + m) * Pnn) / (x2 - 1) // Recurrence formula (5)
        func(n, m, m, r_n, Pnn, Pnn_deriv);
        func(n, -m, m, r_n, Pnn, Pnn_deriv);
        n++; r_n = r_n * r;
        func(n, m, m, r_n, Pnn_next, Pnn_next_deriv);
        func(n, -m, m, r_n, Pnn_next, Pnn_next_deriv);

        P_pre2 = Pnn;
        P_pre1 = Pnn_next;
        for (; n < max_n;) {
            n++; r_n = r_n * r;
            const P_current = ((2 * n - 1) * x * P_pre1 - (n + m - 1) * P_pre2) / (n - m);// Recurrence formula (3)
            const P_deriv = (n * x * P_current - (n + m) * P_pre1) / (x2 - 1) // Recurrence formula (5)
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
        let n = m, r_n = r_m;
        let Pnn_next = x * (2 * m + 1) * Pnn; // Recurrence formula (2)
        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (1 - x2) // Recurrence formula (4)
        func(n, m, m, r_n, Pnn, Pnn_deriv);
        func(n, -m, m, r_n, Pnn, Pnn_deriv);
    }
}

/**
 * Test functions above using "Reparameterization in terms of angles", only for sin(theta)>=0
 *  */
export function CalcALP_Test(theta) {
    console.log(`Test CalcALP for theta=${theta}`)
    const sin = Math.sin, cos = Math.cos;
    const s = sin(theta), c = cos(theta);

    const numExpansions = 5;
    const bufferSize = numExpansions * (numExpansions + 1) / 2;
    const testCase = new Float32Array(bufferSize);
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
    })
    const test1 = CalcALP(5, c);
    const test2 = new Float32Array(bufferSize);
    CalcALP_R(5, c, 1, (n, m, m_abs, r_n, P, P_deriv) => {
        test2[n * (n + 1) / 2 + m_abs] = P;
    });

    const errors1 = VerifyFloatBuffer(testCase, test1);
    const errors2 = VerifyFloatBuffer(testCase, test2);
    [errors1, errors2].forEach(errors => {
        errors.forEach(record => {
            record.n = i2n[record.i];
            record.m = i2m[record.i];
        });
    })
    if (errors1.length != 0) {
        errors1.push(test1)
        console.log("Test for CalcALP():", errors1);
        console.log(Array.from(test1).map((v, i) => { return { n: i2n[i], m: i2m[i], v: v } }));
    }
    if (errors2.length != 0) {
        errors2.push(test2)
        console.log("Test for CalcALP_R():", errors2);
        console.log(Array.from(test2).map((v, i) => { return { n: i2n[i], m: i2m[i], v: v } }));
    }
}

function VerifyFloatBuffer(expect: ArrayLike<number>, data: ArrayLike<number>, max_error = 0.001) {
    function CompareNumber(a: number, b: number, delta = 0.002) {
        return Math.abs(a - b) < delta
    }
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
