pub mod min_sum;
pub mod spa;

use sprs::CsMat;

#[allow(non_snake_case)]
pub struct SyndromeBpCore {
    pub H_csc: CsMat<u8>,
    pub H_csr: CsMat<u8>,
    pub csr_to_csc: Vec<usize>,
    pub channel_llrs: Vec<f64>,
    pub msg_cn_to_vn: Vec<f64>,
    pub msg_vn_to_cn: Vec<f64>,
    pub total_llrs: Vec<f64>,
}

pub trait SyndromeBpStrategy {
    #[allow(non_snake_case)]
    fn new(H: &CsMat<u8>, channel_llrs: &[f64]) -> Self;

    fn vn_update(&mut self);
    fn cn_update(&mut self, s: &[u8]);
    fn total_llrs(&mut self);

    fn get_state(&mut self) -> &mut SyndromeBpCore;
}
