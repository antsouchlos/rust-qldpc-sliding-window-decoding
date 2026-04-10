pub mod bp;

pub trait Decoder {
    fn decode(&mut self, s: &[u8]) -> Vec<u8>;
}

pub trait SoftInitDecoder {
    fn init_soft_info(s: &[f64]);
    fn decode(s: &[f64]) -> Vec<f64>;
}
