use num_traits::Float;

use crate::decoders::{
    Decoder,
    engine::{AccessEngineInternals, BpComputeEngine, ParityCheckMatrix, spa::SpaComputeEngine},
    sliding_window::InnerWindowDecoder,
};

// TODO: Build this in a way were more cores are supported
pub trait BpGdEngine: AccessEngineInternals {}
impl BpGdEngine for SpaComputeEngine {}

#[derive(Clone)]
pub struct VanillaBpSettings {
    pub max_iter: usize,
}

#[derive(Clone)]
pub struct BpGdDecoder<Engine: BpComputeEngine> {
    settings: VanillaBpSettings,
    pcm: ParityCheckMatrix,
    engine: Engine,
    x_hat: Vec<u8>,
    decimated_vns: Vec<usize>,
}

// TODO: Implement this more generally to also support, e.g., SIMD operations
// (not just for floats)
impl<Engine> BpGdDecoder<Engine>
where
    Engine: BpGdEngine,
    Engine::Llr: Float,
{
    fn hard_decision_into(dest: &mut [u8], llrs: &[Engine::Llr]) {
        for (dest_i, &llrs) in dest.iter_mut().zip(llrs) {
            *dest_i = (llrs < -Engine::Llr::neg_zero()) as u8;
        }
    }

    pub fn new(
        settings: VanillaBpSettings,
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
            decimated_vns: Vec::with_capacity(pcm.cols()),
        }
    }

    pub fn reset(&mut self) {
        self.engine.reset();
    }

    fn decimate_next_vn(&mut self) {
        let max_vn = self
            .engine
            .total_llrs()
            .iter()
            .enumerate()
            .filter(|(vn_idx, _)| self.decimated_vns.contains(vn_idx))
            .max_by(|(_, llr1), (_, llr2)| {
                llr1.partial_cmp(llr2).expect("Could not compare llrs")
            })
            .expect("Could not determine maximum total llr");

        self.decimated_vns.push(max_vn.0);
    }
}

impl<Engine> Decoder for BpGdDecoder<Engine>
where
    Engine: BpGdEngine,
    Engine::Llr: Float,
{
    type Settings = VanillaBpSettings;

    fn decode(&mut self, s: &[u8]) -> &[u8] {
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

impl<Engine> InnerWindowDecoder for BpGdDecoder<Engine>
where
    Engine: BpGdEngine,
    Engine::Llr: Float,
{
    type Llr = Engine::Llr;

    fn new(
        settings: Self::Settings,
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
            decimated_vns: Vec::with_capacity(pcm.cols()),
        }
    }

    fn get_cn_to_vn_msg(&self, edge_id: super::engine::EdgeId) -> Self::Llr {
        self.engine.get_cn_to_vn_msg(edge_id)
    }

    fn get_vn_to_cn_msg(&self, edge_id: super::engine::EdgeId) -> Self::Llr {
        self.engine.get_vn_to_cn_msg(edge_id)
    }

    fn get_channel_llr(&self, i: usize) -> Self::Llr {
        self.engine.get_channel_llr(i)
    }

    fn set_cn_to_vn_msg(
        &mut self,
        edge_id: super::engine::EdgeId,
        msg: Self::Llr,
    ) {
        self.engine.set_cn_to_vn_msg(edge_id, msg);
    }

    fn set_vn_to_cn_msg(
        &mut self,
        edge_id: super::engine::EdgeId,
        msg: Self::Llr,
    ) {
        self.engine.set_vn_to_cn_msg(edge_id, msg);
    }

    fn set_channel_llr(&mut self, i: usize, llr: Self::Llr) {
        self.engine.set_channel_llr(i, llr);
    }

    fn reset(&mut self) {
        self.engine.reset();
    }
}
