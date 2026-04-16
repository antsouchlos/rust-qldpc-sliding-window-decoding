pub mod bp;
pub mod bpgd;
pub mod engine;
pub mod meta;

pub trait Decoder {
    type Settings: Clone;
    fn decode(&mut self, s: &[u8]) -> &[u8];
}
