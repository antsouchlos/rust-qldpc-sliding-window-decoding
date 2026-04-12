pub mod bp;
pub mod bpgd;
pub mod core;

pub trait Decoder {
    fn decode(&mut self, s: &[u8]) -> Vec<u8>;
    fn reset(&mut self);
}
