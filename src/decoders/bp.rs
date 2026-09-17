use crate::decoders::{
    Decoder,
    engine::{BpComputeEngine, ParityCheckMatrix},
};

#[derive(Clone)]
pub struct StandardBpSettings<InnerSettings: Clone> {
    pub max_iter: usize,
    pub engine_settings: InnerSettings,
}

#[derive(Clone)]
pub struct StandardBpDecoder<Engine: BpComputeEngine> {
    pub(in crate::decoders) settings: StandardBpSettings<Engine::Settings>,
    pub(in crate::decoders) pcm: ParityCheckMatrix,
    pub(in crate::decoders) engine: Engine,
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
        settings: StandardBpSettings<Engine::Settings>,
        pcm: &ParityCheckMatrix,
        channel_llrs: &[f64],
    ) -> Self {
        let mut engine = Engine::new(pcm, &settings.engine_settings);
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
