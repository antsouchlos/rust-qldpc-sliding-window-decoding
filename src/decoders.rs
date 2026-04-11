pub mod core;
pub mod bp;
pub mod bpgd;

pub trait Decoder {
    fn decode(&mut self, s: &[u8]) -> Vec<u8>;
}
