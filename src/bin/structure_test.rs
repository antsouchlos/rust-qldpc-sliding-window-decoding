use std::ops::Bound;
use std::ops::Range;
use std::ops::RangeBounds;

use num_traits::float::Float;
use sprs::CsMat;

//
//
// Trait definitions
//
//

/// Type used to index edges corresponding to the one-entries in the sparse
/// PCM. Avoids the problem of having to ensure (i,j) pairs point to valid PCM
/// entries
struct EdgeId(u32);

/// Only contains information about the structure of the code. Is able to
/// match (i,j) pairs to EdgeId objects
#[derive(Clone)]
struct ParityCheckMatrix {
    h: CsMat<u8>,
}

impl ParityCheckMatrix {
    fn new(h: CsMat<u8>) -> Self {
        Self { h: h.to_csr() }
    }

    pub fn slice<R1, R2>(
        &self,
        rows: &R1,
        cols: &R2,
    ) -> impl Iterator<Item = EdgeId>
    where
        R1: RangeBounds<usize>,
        R2: RangeBounds<usize>,
    {
        // These bounds are due to the way the edge IDs are generated
        if let Bound::Included(&end) = rows.end_bound() {
            assert!(end < (1 << 16), "row end must be less than 65536");
        }
        if let Bound::Included(&end) = cols.end_bound() {
            assert!(end < (1 << 16), "col end must be less than 65536");
        }

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

    fn get_edge_id(&self, i: usize, j: usize) -> Option<EdgeId> {
        assert!(
            i < (1 << 16) && j < (1 << 16),
            "i and j must be less than 65536"
        );

        if self.h.get(i, j).is_some() {
            Some(EdgeId((i as u32) << 16 | (j as u32)))
        } else {
            None
        }
    }

    fn cols(&self) -> usize {
        self.h.cols()
    }

    fn rows(&self) -> usize {
        self.h.rows()
    }

    fn compute_syndrome(&self, e_hat: &[u8]) -> Vec<u8> {
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
trait BpComputeEngine {
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
trait AccessEngineInternals: BpComputeEngine {
    fn get_cn_to_vn_msg(&self, edge_id: EdgeId) -> Self::Llr;
    fn get_vn_to_cn_msg(&self, edge_id: EdgeId) -> Self::Llr;
    fn get_channel_llr(&self, i: usize) -> Self::Llr;

    fn set_cn_to_vn_msg(&mut self, edge_id: EdgeId, msg: Self::Llr);
    fn set_vn_to_cn_msg(&mut self, edge_id: EdgeId, msg: Self::Llr);
    fn set_channel_llr(&mut self, i: usize, llr: Self::Llr);
}

trait Decoder {
    type Llr: Float;

    fn decode(&mut self, syndrome: &[u8]) -> &[u8];
}

//
//
// Decoder implementations
//
//

#[derive(Clone)]
struct StandardBpSettings {
    max_iter: usize,
}

struct StandardBpDecoder<Engine: BpComputeEngine> {
    settings: StandardBpSettings,
    pcm: ParityCheckMatrix,
    engine: Engine,
    x_hat: Vec<u8>,
}

impl<Engine> StandardBpDecoder<Engine>
where
    Engine: BpComputeEngine,
    Engine::Llr: Float,
{
    fn hard_decision_into(dest: &mut [u8], llrs: &[Engine::Llr]) {
        for (dest_i, &llrs) in dest.iter_mut().zip(llrs) {
            *dest_i = (llrs < -Engine::Llr::neg_zero()) as u8;
        }
    }

    fn new(
        settings: StandardBpSettings,
        pcm: &ParityCheckMatrix,
        channel_llrs: &[Engine::Llr],
    ) -> Self {
        let mut engine = Engine::new(pcm);
        engine.set_channel_llrs(channel_llrs);

        Self {
            settings,
            pcm: pcm.clone(),
            engine,
            x_hat: vec![0u8; pcm.cols()],
        }
    }
}

impl<Engine> Decoder for StandardBpDecoder<Engine>
where
    Engine: BpComputeEngine,
    Engine::Llr: Float,
{
    type Llr = Engine::Llr;

    fn decode(&mut self, s: &[u8]) -> &[u8] {
        self.engine.reset();

        for _ in 0..self.settings.max_iter {
            Self::hard_decision_into(&mut self.x_hat, self.engine.total_llrs());

            if self.pcm.compute_syndrome(&self.x_hat) == s {
                break;
            }

            self.engine.vn_update();
            self.engine.cn_update(s);
        }

        &self.x_hat
    }
}

//
//
// Concrete engine implementations
//
//

#[derive(Clone)]
pub struct Edge {
    pub row: usize,
    pub col: usize,
    pub msg_vn_to_cn: f64,
    pub msg_cn_to_vn: f64,
}

struct MinSumComputeEngine {
    pub edges: Vec<Edge>,
    pub cn_ranges: Vec<Range<usize>>,
    pub vn_indices: Vec<Vec<usize>>,
    pub channel_llrs: Vec<f64>,
    pub total_llrs: Vec<f64>,
    pub num_vns: usize,
    pub num_cns: usize,
}

impl BpComputeEngine for MinSumComputeEngine {
    type Llr = f64;

    fn new(pcm: &ParityCheckMatrix) -> Self {
        let h_csr = pcm.h.to_csr();
        let num_cns = h_csr.rows();
        let num_vns = h_csr.cols();
        let nnz = h_csr.nnz();

        let mut edges = Vec::with_capacity(nnz);
        for (row, row_vec) in h_csr.outer_iterator().enumerate() {
            for (col, _) in row_vec.iter() {
                edges.push(Edge {
                    row,
                    col,
                    msg_vn_to_cn: 0.0,
                    msg_cn_to_vn: 0.0,
                });
            }
        }

        let mut cn_ranges = Vec::with_capacity(num_cns);
        for j in 0..num_cns {
            let start = pcm.h.indptr().index(j);
            let end = pcm.h.indptr().index(j + 1);
            cn_ranges.push(start..end);
        }

        let mut vn_indices = vec![Vec::new(); num_vns];
        for (k, edge) in edges.iter().enumerate() {
            vn_indices[edge.col].push(k);
        }

        Self {
            edges,
            cn_ranges,
            vn_indices,
            channel_llrs: vec![0.0; num_vns],
            total_llrs: vec![0.0; num_vns],
            num_vns,
            num_cns,
        }
    }

    fn set_channel_llrs(&mut self, llrs: &[Self::Llr]) {
        self.channel_llrs.copy_from_slice(llrs);
    }

    fn reset(&mut self) {
        for edge in self.edges.iter_mut() {
            edge.msg_vn_to_cn = 0.0;
            edge.msg_cn_to_vn = 0.0;
        }
    }

    /// Perform check node update using the min-sum approximation.
    ///
    /// To avoid having to do two passes to account for the extrinsic
    /// principle, the two minimum values are found in one pass.
    fn cn_update(&mut self, syndrome: &[u8]) {
        for j in 0..self.num_cns {
            let range = self.cn_ranges[j].clone();
            let syndrome_sign = 1.0 - 2.0 * syndrome[j] as f64;
            let mut total_sign = syndrome_sign;
            let mut min1 = f64::INFINITY;
            let mut min2 = f64::INFINITY;
            let mut min1_idx = range.start;

            // Combine all incoming messages

            for edge_idx in range.clone() {
                let msg = self.edges[edge_idx].msg_vn_to_cn;
                let abs_msg = msg.abs();

                total_sign *= msg.signum();

                if abs_msg < min1 {
                    min2 = min1;
                    min1 = abs_msg;
                    min1_idx = edge_idx;
                } else if abs_msg < min2 {
                    min2 = abs_msg;
                }
            }

            // Assign outgoing messages

            for edge_idx in range {
                let msg = self.edges[edge_idx].msg_vn_to_cn;

                let extrinsic_sign = total_sign * msg.signum();
                let extrinsic_mag =
                    if edge_idx == min1_idx { min2 } else { min1 };

                self.edges[edge_idx].msg_cn_to_vn =
                    extrinsic_sign * extrinsic_mag;
            }
        }
    }

    /// Perform variable node update [1].
    ///
    /// This function also computes the total LLRs.
    ///
    /// [1] H. Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
    ///     with Guided Decimation," arXiv:2312.10950, 2024.
    fn vn_update(&mut self) {
        for i in 0..self.num_vns {
            let num_neighbors = self.vn_indices[i].len();

            // Combine all incoming messages

            self.total_llrs[i] = self.channel_llrs[i];
            for j in 0..num_neighbors {
                let idx = self.vn_indices[i][j];
                self.total_llrs[i] += self.edges[idx].msg_cn_to_vn;
            }

            // Assign outgoing messages

            for j in 0..num_neighbors {
                let edge_idx = self.vn_indices[i][j];
                self.edges[edge_idx].msg_vn_to_cn =
                    self.total_llrs[i] - self.edges[edge_idx].msg_cn_to_vn;
            }
        }
    }

    fn total_llrs(&self) -> &[Self::Llr] {
        &self.total_llrs
    }
}

fn main() {
    println!("Hello, World!");

    let h = todo!();
    let pcm = ParityCheckMatrix::new(h);

    let priors = todo!();

    let decoder = StandardBpDecoder::<MinSumComputeEngine>::new(
        StandardBpSettings { max_iter: 32 },
        &pcm,
        priors,
    );

    let syndrome = todo!();

    let x_hat = decoder.decode(syndrome);
}
