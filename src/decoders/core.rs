pub mod min_sum;
pub mod naive_spa;
pub mod spa;

use std::ops::Range;

use sprs::CsMat;

use crate::decoders::Decoder;

#[derive(Clone)]
pub struct Edge {
    pub row: usize,
    pub col: usize,
    pub msg_vn_to_cn: f64,
    pub msg_cn_to_vn: f64,
}

// TODO: Doc comments to explain data structure
#[derive(Clone)]
pub struct SyndromeBpCore {
    pub edges: Vec<Edge>,
    pub cn_ranges: Vec<Range<usize>>,
    pub vn_indices: Vec<Vec<usize>>,
    pub channel_llrs: Vec<f64>,
    pub total_llrs: Vec<f64>,
    pub num_vns: usize,
    pub num_cns: usize,
}

// TODO: Don't require the new function in this trait. Different strategies
// might need different parameters (e.g., clippling value)
pub trait SyndromeBpStrategy {
    #[allow(non_snake_case)]
    fn new(H: &CsMat<u8>, channel_llrs: &[f64]) -> Self;

    fn vn_update(&mut self);
    fn cn_update(&mut self, s: &[u8]);
    fn total_llrs(&mut self);

    fn get_state(&mut self) -> &mut SyndromeBpCore;
    fn get_state_ref(&self) -> &SyndromeBpCore;
}

pub trait SyndromeBpDecoder: Decoder {
    #[allow(non_snake_case)]
    fn new(
        settings: <Self as Decoder>::Settings,
        H: &CsMat<u8>,
        channel_llrs: &[f64],
    ) -> Self;

    fn reset(&mut self);
}
