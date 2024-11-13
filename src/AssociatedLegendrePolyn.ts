// 


/**
 * Calc Associated Legendre polynomials for given x and p.
 * 
 * @returns {Float32Array} Pnm: P00,P10,P11...
 * <p>Length: p*(p+1)/2</p> 
 * . m<=n.
 */
export function CalcALP(x: number, p: number): Float32Array {
    const sqrt = Math.sqrt;
    let i: number;
    const max_n = p - 1;


    const size = p * (p + 1) / 2;
    const buffer = new Float32Array(size);
    const sinTheta = sqrt(1 - x * x);

    let Pnn = 1; // start from P00
    buffer[0] = Pnn;
    for (let loop = 0; loop < max_n; loop++) { // 最后的Pnn n=max_n被上一轮计算
        let n = loop;
        let m = n;

        let P_pre2 = Pnn;

        n++;
        let P_pre1 = x * (2 * m + 1) * Pnn;
        i = n * (n + 1) / 2 + m; buffer[i] = P_pre1;

        for (; n < max_n;) {
            const P_current = ((2 * n + 1) * x * P_pre1 - (n + m) * P_pre2) / (n - m + 1);
            n++;
            i = n * (n + 1) / 2 + m; buffer[i] = P_current;

            // update preview P
            P_pre2 = P_pre1;
            P_pre1 = P_current;
        }
        Pnn = -(2 * m + 1) * sinTheta * Pnn;
        m++; n = m;// should = loop + 1
        i = n * (n + 1) / 2 + m; buffer[i] = Pnn;
    }
    return buffer;
}