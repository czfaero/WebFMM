// 


/**
 * Calc Associated Legendre polynomials for given x and n.
 * @returns Float32Array: P00, 010, P11, ...Pnm. m<=n.
 */
export function CalcALP(x: number, max_n: number): Float32Array {
    const sqrt = Math.sqrt;
    let i: number;


    const size = (max_n + 2) * (max_n + 1) / 2;
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