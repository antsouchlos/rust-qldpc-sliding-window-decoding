use sprs::CsMat;

use crate::{Decoder, bp_core::SyndromeBpStrategy};

#[allow(non_snake_case)]
pub fn compute_syndrome(H: &CsMat<u8>, e_hat: &[u8]) -> Vec<u8> {
    let mut syndrome = vec![0u8; H.rows()];
    for (row_idx, row) in H.to_csr().outer_iterator().enumerate() {
        for (col_idx, _) in row.iter() {
            syndrome[row_idx] ^= e_hat[col_idx];
        }
    }
    syndrome
}

pub struct Settings {
    pub max_iter: usize,
}

pub struct SyndromeBpDecoder<Core: SyndromeBpStrategy> {
    pub settings: Settings,
    pub core: Core,
}

impl<Core: SyndromeBpStrategy> SyndromeBpDecoder<Core> {
    #[allow(non_snake_case)]
    pub fn new(
        settings: Settings,
        H: &CsMat<u8>,
        channel_llrs: &[f64],
    ) -> Self {
        Self {
            settings,
            core: Core::new(H, channel_llrs),
        }
    }
}

// TODO: Doc comments
impl<Core: SyndromeBpStrategy> Decoder for SyndromeBpDecoder<Core> {
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut e_hat: Vec<u8> = self
            .core
            .get_state()
            .channel_llrs
            .iter()
            .map(|&v| if v < 0.0 { 1 } else { 0 })
            .collect();

        for _ in 0..self.settings.max_iter {
            self.core.vn_update();
            self.core.cn_update(&s);
            self.core.total_llrs();

            e_hat = self
                .core
                .get_state()
                .total_llrs
                .iter()
                .map(|&v| if v < 0.0 { 1 } else { 0 })
                .collect();

            let s_hat = compute_syndrome(&self.core.get_state().H_csc, &e_hat);
            if s_hat == s {
                break;
            }
        }

        e_hat
    }
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;

    #[allow(non_snake_case)]
    fn get_hamming_H() -> CsMat<u8> {
        #[allow(non_snake_case)]
        let mut H = TriMat::<u8>::new((3, 7));

        H.add_triplet(0, 0, 1);
        H.add_triplet(1, 1, 1);
        H.add_triplet(2, 2, 1);
        H.add_triplet(0, 3, 1);
        H.add_triplet(2, 3, 1);
        H.add_triplet(0, 4, 1);
        H.add_triplet(1, 4, 1);
        H.add_triplet(1, 5, 1);
        H.add_triplet(2, 5, 1);
        H.add_triplet(0, 6, 1);
        H.add_triplet(1, 6, 1);
        H.add_triplet(2, 6, 1);

        H.to_csr()
    }

    #[test]
    fn test_compute_syndrome() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();

        let e = vec![0, 1, 0, 1, 0, 1, 0];
        let s = vec![1, 0, 0];

        assert_eq!(s, compute_syndrome(&H, &e));
    }
}
