use crate::decoders::{
    Decoder,
    asced::InnerAscedDecoder,
    engine::{AccessEngineInternals, BpComputeEngine, ParityCheckMatrix},
    sliding_window::InnerWindowDecoder,
};

#[derive(Clone)]
pub struct StandardBpSettings {
    pub max_iter: usize,
}

#[derive(Clone)]
pub struct StandardBpDecoder<Engine: BpComputeEngine> {
    settings: StandardBpSettings,
    pcm: ParityCheckMatrix,
    engine: Engine,
}

impl<Engine> StandardBpDecoder<Engine>
where
    Engine: BpComputeEngine,
{
    fn hard_decision_into(dest: &mut [u8], llrs: &[f64]) {
        for (dest_i, &llrs) in dest.iter_mut().zip(llrs) {
            *dest_i = (llrs < 0f64) as u8;
        }
    }

    pub fn new(
        settings: StandardBpSettings,
        pcm: &ParityCheckMatrix,
        channel_llrs: &[f64],
    ) -> Self {
        let mut engine = Engine::new(pcm);
        engine.set_channel_llrs(channel_llrs);

        Self {
            settings,
            pcm: pcm.clone(),
            engine,
        }
    }

    pub fn reset(&mut self) {
        self.engine.reset();
    }
}

impl<Engine> Decoder for StandardBpDecoder<Engine>
where
    Engine: BpComputeEngine,
{
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut x_hat = vec![0u8; self.pcm.cols()];

        for _ in 0..self.settings.max_iter {
            Self::hard_decision_into(&mut x_hat, &self.engine.total_llrs());

            if self.pcm.compute_syndrome(&x_hat) == s {
                break;
            }

            self.engine.vn_update();
            self.engine.cn_update(s);
        }

        x_hat
    }
}

impl<Engine> InnerWindowDecoder for StandardBpDecoder<Engine>
where
    Engine: AccessEngineInternals,
{
    type Settings = StandardBpSettings;

    fn new(
        settings: StandardBpSettings,
        pcm: &ParityCheckMatrix,
        channel_llrs: &[f64],
    ) -> Self {
        let mut engine = Engine::new(pcm);
        engine.set_channel_llrs(channel_llrs);

        Self {
            settings,
            pcm: pcm.clone(),
            engine,
        }
    }

    fn get_cn_to_vn_msg(&self, edge_id: super::engine::EdgeId) -> f64 {
        self.engine.get_cn_to_vn_msg(edge_id)
    }

    fn get_vn_to_cn_msg(&self, edge_id: super::engine::EdgeId) -> f64 {
        self.engine.get_vn_to_cn_msg(edge_id)
    }

    fn get_channel_llr(&self, i: usize) -> f64 {
        self.engine.get_channel_llr(i)
    }

    fn set_cn_to_vn_msg(&mut self, edge_id: super::engine::EdgeId, msg: f64) {
        self.engine.set_cn_to_vn_msg(edge_id, msg);
    }

    fn set_vn_to_cn_msg(&mut self, edge_id: super::engine::EdgeId, msg: f64) {
        self.engine.set_vn_to_cn_msg(edge_id, msg);
    }

    fn set_channel_llr(&mut self, i: usize, llr: f64) {
        self.engine.set_channel_llr(i, llr);
    }

    fn reset(&mut self) {
        self.engine.reset();
    }
}

impl<Engine> InnerAscedDecoder for StandardBpDecoder<Engine>
where
    Engine: BpComputeEngine,
{
    type Settings = StandardBpSettings;

    fn new(
        settings: Self::Settings,
        pcm: &sprs::CsMat<u8>,
        channel_llrs: &[f64],
    ) -> Self {
        Self::new(settings, &ParityCheckMatrix::new(pcm), channel_llrs)
    }

    fn reset(&mut self) {
        self.engine.reset();
    }
}
