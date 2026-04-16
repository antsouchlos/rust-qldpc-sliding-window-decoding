use num_traits::Float;

use crate::decoders::{
    Decoder,
    core::{BpComputeEngine, ParityCheckMatrix},
};

#[derive(Clone)]
pub struct VanillaBpSettings {
    pub max_iter: usize,
}

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
