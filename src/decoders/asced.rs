use std::collections::HashSet;

use sprs::CsMat;

use crate::decoders::{Decoder, engine::ParityCheckMatrix};
use rand::{Rng, RngExt, SeedableRng};

/// [2] L. Wursthorn, et al., "Affine Subcode Ensemble Decoding for
///     Degeneracy-Aware Quantum Error Correction," arXiv:2605.06547, 2026.

/// This function implements algorithm 1 from [1]: It generates additional
/// rows for a PCM that don't create 4-cycles.
///
/// Returns `(new_row_indices, new_feasible_set)`.
///
/// # References
///
/// [1] J. Mandelbaum, H. Jäkel, L. Schmalen, "Subcode Ensemble Decoding of
///     Linear Block Codes," 2025 IEEE International Symposium on Information
///     Theory (ISIT), 2025.
///
fn generate_4_cycle_free_row(
    h: &CsMat<u8>,
    row_weight: usize,
    mut feasible_set: Vec<usize>,
    rng: &mut impl Rng,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut new_row_indices = Vec::<usize>::with_capacity(row_weight);

    while new_row_indices.len() < row_weight {
        // Line 6
        if feasible_set.len() == 0 {
            return None;
        }

        // Line 8
        let pick = rng.random_range(0..feasible_set.len());
        let i_new = feasible_set.swap_remove(pick);

        new_row_indices.push(i_new);

        // Line 10
        feasible_set.retain(|&l| {
            (0..h.rows())
                .all(|j| !(h.get(j, l).is_some() && h.get(j, i_new).is_some()))
        });
    }

    Some((new_row_indices, feasible_set))
}

/// Generate a vector of indices, each index corresponing to another 1-entry of
/// a splitter
fn generate_splitter_indices(
    h: &CsMat<u8>,
    row_weight: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    for _ in 0..100 {
        let feasible_set: Vec<usize> = (0..h.cols()).collect();

        if let Some(indices) =
            generate_4_cycle_free_row(h, row_weight, feasible_set, rng)
        {
            return indices.0;
        }
    }

    panic!("Splitter generation failed")
}

fn append_splitters_to_pcm(
    h: &CsMat<u8>,
    splitters: &[Vec<usize>],
) -> CsMat<u8> {
    let m_new = h.rows() + splitters.len();
    let n_new = h.cols();

    let mut triplet_matrix = sprs::TriMat::new((m_new, n_new));

    for (j, row) in h.outer_iterator().enumerate() {
        for (i, &val) in row.iter() {
            triplet_matrix.add_triplet(j, i, val);
        }
    }

    for (new_row_idx, new_row) in splitters.iter().enumerate() {
        let j = h.rows() + new_row_idx;

        for &i in new_row {
            triplet_matrix.add_triplet(j, i, 1u8);
        }
    }

    triplet_matrix.to_csr()
}

/// Wraps an InnerAscedDecoder and handles extending the syndrome by g before
/// decoding
struct InnerDecoderWrapper<InnerDecoder>
where
    InnerDecoder: InnerAscedDecoder,
{
    decoder: InnerDecoder,
    g: Vec<u8>,
}

impl<InnerDecoder> Decoder for InnerDecoderWrapper<InnerDecoder>
where
    InnerDecoder: InnerAscedDecoder,
{
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut extended = s.to_vec();
        extended.extend_from_slice(&self.g);

        self.decoder.decode(&extended)
    }
}

pub trait InnerAscedDecoder: Decoder {
    type Settings: Clone;

    fn new(
        settings: Self::Settings,
        pcm: &CsMat<u8>,
        channel_llrs: &[f64],
    ) -> Self;
}

struct AscedSettings {
    /// Hamming weight of a splitter row
    splitter_weight: usize,
    /// Number of batches, i.e., sets of differing splitters
    num_batches: usize,
    /// Number of splitters to add to each batch
    delta: usize,
}

pub struct AscedDecoder<InnerDecoder>
where
    InnerDecoder: InnerAscedDecoder,
{
    decoder_ensemble: Vec<InnerDecoderWrapper<InnerDecoder>>,
    pcm: ParityCheckMatrix,
}

impl<InnerDecoder> AscedDecoder<InnerDecoder>
where
    InnerDecoder: InnerAscedDecoder,
{
    fn new(
        settings: AscedSettings,
        inner_settings: InnerDecoder::Settings,
        h: &CsMat<u8>,
        channel_llrs: &[f64],
    ) -> Self {
        let mut rng = rand::rngs::StdRng::try_from_rng(&mut rand::rngs::SysRng)
            .expect("Failed to create rng");

        let batch_size = 1 << settings.delta;
        let total_paths = settings.num_batches * batch_size;

        let mut decoder_ensemble: Vec<InnerDecoderWrapper<InnerDecoder>> =
            Vec::with_capacity(total_paths);
        let mut splitters: HashSet<Vec<Vec<usize>>> = HashSet::new();

        for _ in 0..settings.num_batches {
            //
            // Generate splitters and build extended PCM
            //

            let batch_splitters = {
                let mut result = None;

                for _ in 0..100 {
                    let mut splitters_candidate: Vec<Vec<usize>> =
                        Vec::with_capacity(settings.delta);

                    for _ in 0..settings.delta {
                        let growing_pcm =
                            append_splitters_to_pcm(&h, &splitters_candidate);
                        let row = generate_splitter_indices(
                            &growing_pcm,
                            settings.splitter_weight,
                            &mut rng,
                        );
                        splitters_candidate.push(row);
                    }

                    // Check if the generated set of splitters already exists
                    // for some other batch.
                    // In order to make the sets of splitters comparable, they
                    // are first sorted.

                    let mut sorted_splitters = splitters_candidate.clone();
                    for row in &mut sorted_splitters {
                        row.sort(); // `row` contains the indices of the `1`
                    }
                    sorted_splitters.sort();

                    if splitters.insert(sorted_splitters) {
                        result = Some(splitters_candidate);
                        break;
                    }
                }
                result.expect(
                    "Failed to generate unique splitter set after 100 attempts",
                )
            };

            let extended_pcm = append_splitters_to_pcm(&h, &batch_splitters);

            //
            // Generate syndrome bit guesses
            //

            for g_idx in 0..batch_size {
                let g: Vec<u8> = (0..settings.delta)
                    .map(|r| ((g_idx >> r) & 1) as u8)
                    .collect();

                decoder_ensemble.push(InnerDecoderWrapper {
                    decoder: InnerDecoder::new(
                        inner_settings.clone(),
                        &extended_pcm,
                        channel_llrs,
                    ),
                    g,
                });
            }
        }

        Self {
            decoder_ensemble,
            pcm: ParityCheckMatrix::new(h),
        }
    }
}

impl<InnerDecoder> Decoder for AscedDecoder<InnerDecoder>
where
    InnerDecoder: InnerAscedDecoder,
{
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut decoders = self.decoder_ensemble.iter_mut();

        // Fallback in case no estimate passes the syndrome check
        let first_estimate = decoders.next().map(|d| d.decode(s)).unwrap();

        decoders
            .map(|decoder| decoder.decode(s))
            .chain(std::iter::once(first_estimate.clone()))
            .filter(|estimate| self.pcm.compute_syndrome(estimate) == s)
            .min_by_key(|estimate| estimate.iter().filter(|&&b| b != 0).count())
            .unwrap_or(first_estimate)
    }
}
