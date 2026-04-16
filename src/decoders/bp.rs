use num_traits::Float;

use crate::decoders::{
    Decoder,
    engine::{AccessEngineInternals, BpComputeEngine, ParityCheckMatrix},
    meta::sliding_window::InnerWindowDecoder,
};

#[derive(Clone)]
pub struct VanillaBpSettings {
    pub max_iter: usize,
}

#[derive(Clone)]
pub struct VanillaBpDecoder<Engine: BpComputeEngine> {
    settings: VanillaBpSettings,
    pcm: ParityCheckMatrix,
    engine: Engine,
    x_hat: Vec<u8>,
}

// TODO: Implement this more generally to also support, e.g., SIMD operations
// (not just for floats)
impl<Engine> VanillaBpDecoder<Engine>
where
    Engine: BpComputeEngine,
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
        }
    }

    pub fn reset(&mut self) {
        self.engine.reset();
    }
}

impl<Engine> Decoder for VanillaBpDecoder<Engine>
where
    Engine: BpComputeEngine,
    Engine::Llr: Float,
{
    type Settings = VanillaBpSettings;

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

impl<Engine> InnerWindowDecoder for VanillaBpDecoder<Engine>
where
    Engine: BpComputeEngine,
    Engine: AccessEngineInternals,
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
