pub mod bp;
pub mod bpgd;
pub mod core;

pub trait Decoder {
    type Settings: Clone;
    fn decode(&mut self, s: &[u8]) -> Vec<u8>;
}
