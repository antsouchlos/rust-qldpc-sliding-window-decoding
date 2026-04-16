use rust_qldpc::decoders::{
    Decoder,
    vanilla_bp::{VanillaBpDecoder, VanillaBpSettings},
    core::{ParityCheckMatrix, spa::SpaComputeEngine},
};
use sprs::TriMat;

fn make_ldpc(n: usize) -> sprs::CsMat<u8> {
    let mut m = TriMat::<u8>::new((n, 2 * n));

    let shifts0 = [0usize, n / 7 + 1, 2 * n / 7 + 1];
    for shift in shifts0 {
        for i in 0..n {
            m.add_triplet(i, (i + shift) % n, 1);
        }
    }

    let shifts1 = [0usize, n / 5 + 1, 3 * n / 7 + 1];
    for shift in shifts1 {
        for i in 0..n {
            m.add_triplet(i, n + (i + shift) % n, 1);
        }
    }

    m.to_csr()
}

fn main() {
    let n = 1000;
    let h = make_ldpc(n);
    let channel_llrs = vec![2.0f64; h.cols()];
    let mut syndrome = vec![0u8; h.rows()];
    syndrome[0] = 1;

    let pcm = ParityCheckMatrix::new(h);

    let settings = VanillaBpSettings { max_iter: 100 };

    let mut decoder = VanillaBpDecoder::<SpaComputeEngine>::new(
        settings,
        &pcm,
        &channel_llrs,
    );

    for _ in 0..1000 {
        decoder.reset();
        decoder.decode(&syndrome);
    }
}
