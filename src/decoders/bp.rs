use crate::decoders::{
    Decoder,
    core::{Edge, SyndromeBpDecoder, SyndromeBpStrategy},
};

pub fn compute_syndrome(
    edges: &[Edge],
    num_cns: usize,
    e_hat: &[u8],
) -> Vec<u8> {
    let mut syndrome = vec![0u8; num_cns];

    for edge in edges {
        syndrome[edge.row] ^= e_hat[edge.col];
    }

    syndrome
}

#[derive(Clone)]
pub struct Settings {
    pub max_iter: usize,
}

#[derive(Clone)]
pub struct SimpleSyndromeBpDecoder<Core: SyndromeBpStrategy> {
    pub settings: Settings,
    pub core: Core,
}

impl<Core: SyndromeBpStrategy> Decoder for SimpleSyndromeBpDecoder<Core> {
    type Settings = Settings;

    // TODO: Should a reset happen here?
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

            let state = self.core.get_state();
            let s_hat = compute_syndrome(&state.edges, state.num_cns, &e_hat);
            if s_hat == s {
                break;
            }
        }

        e_hat
    }
}

impl<Core: SyndromeBpStrategy> SyndromeBpDecoder
    for SimpleSyndromeBpDecoder<Core>
{
    #[allow(non_snake_case)]
    fn new(
        settings: <Self as Decoder>::Settings,
        H: &sprs::CsMat<u8>,
        channel_llrs: &[f64],
    ) -> Self {
        Self {
            settings: settings,
            core: Core::new(H, channel_llrs),
        }
    }

    fn reset(&mut self) {
        for edge in &mut self.core.get_state().edges {
            edge.msg_vn_to_cn = 0.0;
            edge.msg_cn_to_vn = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;

    #[allow(non_snake_case)]
    fn get_hamming_H() -> sprs::CsMat<u8> {
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

        // Build edges from H manually to test compute_syndrome
        let mut edges = Vec::new();
        for (row, row_vec) in H.outer_iterator().enumerate() {
            for (col, _) in row_vec.iter() {
                edges.push(Edge {
                    row,
                    col,
                    msg_vn_to_cn: 0.0,
                    msg_cn_to_vn: 0.0,
                });
            }
        }

        let e = vec![0, 1, 0, 1, 0, 1, 0];
        let s = vec![1, 0, 0];

        assert_eq!(s, compute_syndrome(&edges, H.rows(), &e));
    }
}
