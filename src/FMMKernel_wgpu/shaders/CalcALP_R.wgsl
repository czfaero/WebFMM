// Constants: numExpansions, eps
// Need define: CalcALP_R_callback
/**
 * Calcuate something about X=(r,theta,phi)   
 * As a piece of data is calc, callback will be called with:  
 * n:  0 <= n < numExpansions  
 * m: -n <= m <= n  
 * m_abs: abs(m)  
 * r_n : r^n  
 * p : Associated Legendre polynomials for n, m at x=cos(theta).  
 * p_d: Derivative of Associated Legendre polynomials. By theta at x. 
 * @param theta Spherical coordinate, 0<= theta <= PI
 * @param r Spherical coordinate
 * @param {function} func callback function
 */
fn CalcALP_R(theta: f32, r: f32) {
    var i: u32;
    let max_n = numExpansions - 1;

    let x = cos(theta);
    var sinTheta = sin(theta);


    if (sinTheta < eps) {
        sinTheta = eps;
    }

    var Pnn: f32 = 1; // start from P00
    var r_m: f32 = 1; // r^m
    var m: i32 = 0;
    var P_pre2: f32;
    var P_pre1: f32;

    // for m=0, call func only once
    {
        var n: i32 = 0;
        var r_n = r_m;

        n++; r_n = r_n * r;

        let Pnn_next = x * (2 * m + 1) * Pnn; // Recurrence formula (2)

        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (sinTheta); // Recurrence formula (4)
        let Pnn_next_deriv = (n * x * Pnn_next - (n + m) * Pnn) / sinTheta; // Recurrence formula (5)

        CalcALP_R_callback(0, m, m, r_m, Pnn, Pnn_deriv);
        CalcALP_R_callback(n, m, m, r_n, Pnn_next, Pnn_next_deriv);

        P_pre2 = Pnn;
        P_pre1 = Pnn_next;
        for (; n < max_n;) {
            n++; r_n = r_n * r;
            let P_current = ((2 * n - 1) * x * P_pre1 - (n + m - 1) * P_pre2) / (n - m);// Recurrence formula (3)
            let P_deriv = (n * x * P_current - (n + m) * P_pre1) / (sinTheta); // Recurrence formula (5)
            CalcALP_R_callback(n, m, m, r_n, P_current, P_deriv);
            P_pre2 = P_pre1;
            P_pre1 = P_current;
        }

    }
    // start from m++ -> 1
    for (; m < max_n - 1;) {
        Pnn = -(2 * m + 1) * sinTheta * Pnn;// Recurrence formula (1)
        m++; r_m = r_m * r;
        var n = m, r_n = r_m;
        let Pnn_next = x * (2 * m + 1) * Pnn; // Recurrence formula (2)
        n++;
        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (sinTheta); // Recurrence formula (4)
        let Pnn_next_deriv = (n * x * Pnn_next - (n + m) * Pnn) / (sinTheta); // Recurrence formula (5)
        CalcALP_R_callback(n - 1, m, m, r_n, Pnn, Pnn_deriv);
        CalcALP_R_callback(n - 1, -m, m, r_n, Pnn, Pnn_deriv);
        r_n = r_n * r;
        CalcALP_R_callback(n, m, m, r_n, Pnn_next, Pnn_next_deriv);
        CalcALP_R_callback(n, -m, m, r_n, Pnn_next, Pnn_next_deriv);

        P_pre2 = Pnn;
        P_pre1 = Pnn_next;
        for (; n < max_n;) {
            n++; r_n = r_n * r;
            let P_current = ((2 * n - 1) * x * P_pre1 - (n + m - 1) * P_pre2) / (n - m);// Recurrence formula (3)
            let P_deriv = (n * x * P_current - (n + m) * P_pre1) / (sinTheta); // Recurrence formula (5)
            CalcALP_R_callback(n, m, m, r_n, P_current, P_deriv);
            CalcALP_R_callback(n, -m, m, r_n, P_current, P_deriv);
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
        let Pnn_deriv = ((n - m) * Pnn_next - n * x * Pnn) / (sinTheta); // Recurrence formula (4)
        CalcALP_R_callback(n - 1, m, m, r_m, Pnn, Pnn_deriv);
        CalcALP_R_callback(n - 1, -m, m, r_m, Pnn, Pnn_deriv);
    }
}


