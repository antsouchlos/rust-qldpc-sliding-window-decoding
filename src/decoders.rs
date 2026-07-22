pub mod bp;
pub mod engine;
pub mod sliding_window;

pub trait Decoder {
    fn decode(&mut self, s: &[u8]) -> Vec<u8>;
}
