pub mod min_sum;
pub mod naive_spa;
pub mod spa;

use std::ops::RangeBounds;

use sprs::CsMat;

/// Type used to index edges corresponding to the one-entries in the sparse
/// PCM. Avoids the problem of having to ensure (i,j) pairs point to valid PCM
/// entries
#[derive(Clone, Copy)]
pub struct EdgeId(u32);

/// Only contains information about the structure of the code. Is able to
/// match (i,j) pairs to EdgeId objects
#[derive(Clone)]
pub struct ParityCheckMatrix {
    h: CsMat<u8>,
}

// TODO: Rethink if having this makes any sense at all
impl ParityCheckMatrix {
    pub fn new(h: &CsMat<u8>) -> Self {
        Self { h: h.to_csr() }
    }

    // TODO: Ensure that rows and cols are within 0..65536
    pub fn slice<R1, R2>(
        &self,
        rows: &R1,
        cols: &R2,
    ) -> impl Iterator<Item = EdgeId>
    where
        R1: RangeBounds<usize>,
        R2: RangeBounds<usize>,
    {
        self.h
            .outer_iterator()
            .enumerate()
            .filter(|(j, _)| rows.contains(j))
            .flat_map(|(j, row)| {
                row.iter()
                    .filter(|(i, _)| cols.contains(i))
                    .map(move |(i, _)| EdgeId((j as u32) << 16 | (i as u32)))
                    .collect::<Vec<_>>()
            })
    }

    pub fn get_edge_id(&self, j: usize, i: usize) -> Option<EdgeId> {
        assert!(
            j < (1 << 16) && i < (1 << 16),
            "i and j must be less than 65536"
        );

        if self.h.get(j, i).is_some() {
            Some(EdgeId((j as u32) << 16 | (i as u32)))
        } else {
            None
        }
    }

    pub fn cols(&self) -> usize {
        self.h.cols()
    }

    pub fn rows(&self) -> usize {
        self.h.rows()
    }

    pub fn compute_syndrome(&self, e_hat: &[u8]) -> Vec<u8> {
        assert!(e_hat.len() == self.cols());
        assert!(e_hat.iter().all(|&val| val == 0 || val == 1));

        let mut syndrome = vec![0u8; self.rows()];

        for (j, row) in self.h.outer_iterator().enumerate() {
            let mut parity = 0u8;

            for (i, &val) in row.iter() {
                parity ^= e_hat[i] * val;
            }

            syndrome[j] = parity;
        }
        syndrome
    }
}

/// (Only) responsible for actual computation
pub trait BpComputeEngine {
    type Llr: Copy;

    fn new(pcm: &ParityCheckMatrix) -> Self;
    fn set_channel_llrs(&mut self, llrs: &[Self::Llr]);
    fn reset(&mut self);

    fn cn_update(&mut self, syndrome: &[u8]);
    fn vn_update(&mut self);
    fn total_llrs(&self) -> &[Self::Llr];
}

/// Some decoders need access to the engine internals during the decoding
/// process. This trait gives them access without exposing the actual
/// internal structure.
pub trait AccessEngineInternals: BpComputeEngine {
    fn get_cn_to_vn_msg(&self, edge_id: EdgeId) -> Self::Llr;
    fn get_vn_to_cn_msg(&self, edge_id: EdgeId) -> Self::Llr;
    fn get_channel_llr(&self, i: usize) -> Self::Llr;

    fn set_cn_to_vn_msg(&mut self, edge_id: EdgeId, msg: Self::Llr);
    fn set_vn_to_cn_msg(&mut self, edge_id: EdgeId, msg: Self::Llr);
    fn set_channel_llr(&mut self, i: usize, llr: Self::Llr);
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;

    fn get_hamming_h() -> sprs::CsMat<u8> {
        let mut h = TriMat::<u8>::new((3, 7));

        h.add_triplet(0, 0, 1);
        h.add_triplet(1, 1, 1);
        h.add_triplet(2, 2, 1);
        h.add_triplet(0, 3, 1);
        h.add_triplet(2, 3, 1);
        h.add_triplet(0, 4, 1);
        h.add_triplet(1, 4, 1);
        h.add_triplet(1, 5, 1);
        h.add_triplet(2, 5, 1);
        h.add_triplet(0, 6, 1);
        h.add_triplet(1, 6, 1);
        h.add_triplet(2, 6, 1);

        h.to_csr()
    }

    #[test]
    fn test_compute_syndrome() {
        let h = get_hamming_h();
        let pcm = ParityCheckMatrix::new(&h);

        let e_hat = [1, 0, 0, 0, 0, 0, 0];
        let syndrome = pcm.compute_syndrome(&e_hat);
        assert_eq!(syndrome, [1, 0, 0]);

        let e_hat = [0, 1, 0, 0, 0, 0, 0];
        let syndrome = pcm.compute_syndrome(&e_hat);
        assert_eq!(syndrome, [0, 1, 0]);

        let e_hat = [0, 0, 1, 0, 0, 0, 0];
        let syndrome = pcm.compute_syndrome(&e_hat);
        assert_eq!(syndrome, [0, 0, 1]);
    }
}
