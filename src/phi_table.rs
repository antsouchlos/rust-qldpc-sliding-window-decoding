/// Lookup table for phi(x) = -ln(tanh(x/2)), evaluated via linear interpolation.
///
/// phi is used in the log-domain sum-product check-node update. It has the
/// property that phi(phi(x)) = x (self-inverse), so the same table handles
/// both the forward and inverse transforms.
///
/// Table covers x ∈ [0, x_max]. For x ≥ x_max, phi(x) ≈ 0 (returned as 0.0).
/// For x = 0, phi(x) → ∞ (clamped to x_max at the first table entry).
pub struct PhiTable {
    table: Vec<f64>,
    dx_inv: f64,
    x_max: f64,
}

impl PhiTable {
    pub fn new(n: usize, x_max: f64) -> Self {
        let dx = x_max / (n - 1) as f64;
        let table = (0..n)
            .map(|i| {
                let x = i as f64 * dx;
                if x < 1e-10 {
                    x_max
                } else {
                    -(x / 2.0).tanh().ln()
                }
            })
            .collect();
        PhiTable { table, dx_inv: 1.0 / dx, x_max }
    }

    /// Look up phi(x) using linear interpolation between table entries.
    #[inline]
    pub fn lookup(&self, x: f64) -> f64 {
        if x >= self.x_max {
            return 0.0;
        }
        let idx = x * self.dx_inv;
        let i = idx as usize;
        let frac = idx - i as f64;
        let i1 = (i + 1).min(self.table.len() - 1);
        self.table[i] + frac * (self.table[i1] - self.table[i])
    }
}

impl Default for PhiTable {
    fn default() -> Self {
        Self::new(8192, 20.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_self_inverse() {
        let table = PhiTable::default();
        // Tolerance is looser for large x because phi(x) is tiny (≈ 2e^{-x}) and
        // falls in the steep near-zero region of the table where phi'' is large.
        for &(x, tol) in &[(0.5f64, 1e-4), (1.0, 1e-4), (2.0, 1e-3), (3.0, 1e-2), (5.0, 5e-2)] {
            let y = table.lookup(x);
            let z = table.lookup(y);
            assert!((z - x).abs() < tol, "phi(phi({x})) = {z}, expected {x} (tol {tol})");
        }
    }

    #[test]
    fn test_phi_accuracy() {
        let table = PhiTable::default();
        for &x in &[0.5f64, 1.0, 2.0, 5.0] {
            let exact = -(x / 2.0).tanh().ln();
            let approx = table.lookup(x);
            assert!((approx - exact).abs() < 1e-5, "phi({x}): got {approx}, exact {exact}");
        }
    }

    #[test]
    fn test_phi_boundary() {
        let table = PhiTable::default();
        assert_eq!(table.lookup(20.0), 0.0);
        assert_eq!(table.lookup(100.0), 0.0);
    }
}
