
// Need define var<workgroup> Pnm: array<f32, PnmSize>; //p * (p + 1) / 2
/**
 * Calc Associated Legendre polynomials for given x=cos(theta) and p.  
 * Set Buffer Pnm: P00,P10,P11...  
 * 0<=n<p, 0<=m<=n  
 * Size: p*(p+1)/2
 */
fn CalcALP(p: u32, x: f32) {

    var i: u32;
    let max_n = p - 1;


    //let size = p * (p + 1) / 2;
    //let buffer = new Float32Array(size);
    let sinTheta = sqrt(1 - x * x);

    var Pnn: f32 = 1; // start from P00
    Pnm[0] = Pnn;
    for (var m: u32 = 0; m < max_n;) { // 最后的Pnn n=max_n被上一轮计算
        var n = m;

        var P_pre2 = Pnn;

        n++;
        var P_pre1 = x * f32(2 * m + 1) * Pnn;// Recurrence formula (2)
        i = n * (n + 1) / 2 + m; Pnm[i] = P_pre1;

        for (; n < max_n;) {
            let P_current = (f32(2 * n + 1) * x * P_pre1 - f32(n + m) * P_pre2) / f32(n - m + 1);// Recurrence formula (3)
            n++;
            i = n * (n + 1) / 2 + m; Pnm[i] = P_current;

            // update preview P
            P_pre2 = P_pre1;
            P_pre1 = P_current;
        }
        Pnn = -f32(2 * m + 1) * sinTheta * Pnn;// Recurrence formula (1)
        m++; n = m;// should = loop + 1
        i = n * (n + 1) / 2 + m; Pnm[i] = Pnn;
    }
}
