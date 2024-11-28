// 


/**
 * Calc Associated Legendre polynomials for given x and p.  
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
    for (let m = 0; m < max_n; m++) { // 最后的Pnn n=max_n被上一轮计算
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