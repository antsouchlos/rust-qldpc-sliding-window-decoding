use crate::Decoder;
use crate::bp_core::{SyndromeBpDecoderCore, compute_syndrome};

#[derive(PartialEq)]
pub enum BpMethod {
    Spa,
    MinSum,
}

pub struct Settings {
    pub max_iter: usize,
    pub bp_method: BpMethod,
}

struct SyndromeBpDecoder {
    settings: Settings,
    core: SyndromeBpDecoderCore,
}

// TODO: Doc comments
impl Decoder for SyndromeBpDecoder {
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut e_hat: Vec<u8> = self
            .core
            .channel_llrs
            .iter()
            .map(|&v| if v < 0.0 { 1 } else { 0 })
            .collect();

        for _ in 0..self.settings.max_iter {
            self.core.vn_update();
            if self.settings.bp_method == BpMethod::Spa {
                self.core.cn_update_spa(s);
            } else if self.settings.bp_method == BpMethod::MinSum {
                self.core.cn_update_min_sum(s);
            }
            self.core.total_llrs();

            e_hat = self
                .core
                .total_llrs
                .iter()
                .map(|&v| if v < 0.0 { 1 } else { 0 })
                .collect();

            let s_hat = compute_syndrome(&self.core.H_csc, &e_hat);
            if s_hat == s {
                break;
            }
        }

        e_hat
    }
}
