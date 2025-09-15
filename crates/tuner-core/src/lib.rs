use std::collections::VecDeque;
use std::sync::mpsc;

pub const FFT_INPUT_SIZE: usize = 8192; // (48000 / 20 as usize).next_power_of_two()
pub const FFT_OUTPUT_SIZE: usize = FFT_INPUT_SIZE / 2;

pub type RawChunkedPacket = Vec<f32>;
pub type CorrectedChunkedPacket = [f32; FFT_INPUT_SIZE];
pub type DFFTPacket = [f32; FFT_OUTPUT_SIZE]; // cut off Nyquist

pub trait SizedArray {
    const LEN: usize;
}

impl<const N: usize, T> SizedArray for [T; N] {
    const LEN: usize = N;
}

#[allow(dead_code)]
pub fn dfft(chunk: CorrectedChunkedPacket) -> DFFTPacket {
    let mut output = [0.0; FFT_OUTPUT_SIZE];
    // Use the total number of time-domain samples for the DFT length.
    const N: usize = FFT_INPUT_SIZE;
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for (n, chunk_val) in chunk.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / N as f32;
            let (s, c) = angle.sin_cos();
            sum_re += chunk_val * c;
            sum_im += chunk_val * s;
        }
        *out = (sum_re * sum_re + sum_im * sum_im).sqrt();
    }
    output
}

#[allow(clippy::assertions_on_constants)]
pub fn fast_dfft(chunk: CorrectedChunkedPacket) -> DFFTPacket {
    use rustfft::{num_complex::Complex, Fft};
    use std::sync::{Arc, LazyLock};
    debug_assert!(CorrectedChunkedPacket::LEN.is_power_of_two());
    static FFT_PLAN: LazyLock<Arc<dyn Fft<f32>>> = LazyLock::new(|| {
        let mut planner = rustfft::FftPlanner::<f32>::new();
        planner.plan_fft_forward(CorrectedChunkedPacket::LEN)
    });
    let mut buf = chunk.map(|x| Complex::new(x, 0.0));
    FFT_PLAN.process(&mut buf);
    let whole_buf = buf.map(|x| x.norm());
    assert!(DFFTPacket::LEN * 2 == CorrectedChunkedPacket::LEN);
    // SAFETY: whole_buf is a slice of length DFFTPacket::LEN * 2, so it can be cast to a slice of length DFFTPacket::LEN
    let [result, _] = unsafe { *(whole_buf.as_ptr() as *const [DFFTPacket; 2]) };
    result
}

// Compute the dominant frequency (in Hz) from a magnitude spectrum.
// Returns 0.0 if all values are non-finite (should not happen for normal inputs).
pub fn max_freq(spectrum: &[f32], sample_rate: f32) -> f32 {
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in spectrum.iter().enumerate() {
        if v.is_finite() && v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    let n = spectrum.len() as f32;
    (max_idx as f32) * (sample_rate / n)
}

// Useful note-related constants
pub const A4_FREQ: f32 = 440.0;
pub const A4_NOTE_NUM: i32 = 69;
pub const A4_OCTAVE: i32 = 4;
// Returns MIDI note number, + fractional detune (in cents)
pub fn freq_to_note(freq: f32) -> (i32, f32) {
    let a4_diff = 12.0 * (freq / A4_FREQ).log2();
    let note = a4_diff.round() as i32;
    (A4_NOTE_NUM + note, (a4_diff - note as f32) * 100.0)
}
// Note number to human-readable note name
pub fn note_to_name(note: i32) -> String {
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let note_diff_c4 = note - A4_NOTE_NUM + 9;
    let note_idx = note_diff_c4.rem_euclid(12) as usize;
    let octave = (note_diff_c4.div_euclid(12)) + A4_OCTAVE;
    format!("{}{}", note_names[note_idx], octave)
}

pub fn correct_chunk_size(
    rx: mpsc::Receiver<RawChunkedPacket>,
) -> mpsc::Receiver<CorrectedChunkedPacket> {
    let (tx, ret_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buffer = VecDeque::new();
        for chunk in rx {
            buffer.extend(chunk);
            while buffer.len() >= FFT_INPUT_SIZE {
                let sized_chunk: CorrectedChunkedPacket = buffer
                    .drain(0..FFT_INPUT_SIZE)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                tx.send(sized_chunk).unwrap();
            }
        }
    });
    ret_rx
}

#[cfg(test)]
mod tests {
    use super::*;

    mod notes {
        use super::*;

        #[test]
        fn note_names() {
            assert_eq!(note_to_name(69), "A4");
            assert_eq!(note_to_name(70), "A#4");
            assert_eq!(note_to_name(71), "B4");
            assert_eq!(note_to_name(72), "C5");
            assert_eq!(note_to_name(60), "C4");
            assert_eq!(note_to_name(59), "B3");
        }

        #[test]
        fn note_freqs() {
            let simple_octave_tests = [
                (440.0, (69, 0.0)),
                (880.0, (69 + 12, 0.0)),
                (220.0, (69 - 12, 0.0)),
            ];
            let semitone_tests = [
                (4186.0, (108, 0.0)), // C8
                (1864.7, (94, 0.0)),  // A#6
            ];
            let detune_tests = [(441.0, (69, 4.0)), (445.0, (69, 20.0))];

            let all_tests = simple_octave_tests
                .into_iter()
                .chain(semitone_tests.into_iter())
                .chain(detune_tests.into_iter());

            for (freq, (exp_note, exp_detune)) in all_tests {
                let (note, detune) = freq_to_note(freq);
                assert_eq!(note, exp_note);
                assert!((detune - exp_detune).abs() < 0.5);
            }
        }
    }

    mod fft {
        use super::*;

        #[test]
        fn dfft_delta_is_flat_one() {
            // x[n] = delta[n] -> X[k] = 1 for all k (unnormalized DFT)
            let mut x = [0.0f32; FFT_INPUT_SIZE];
            x[0] = 1.0;
            let spectrum = dfft(x);
            for &v in spectrum.iter() {
                assert!((v - 1.0).abs() < 1e-4, "expected ~1.0, got {}", v);
            }
        }

        #[test]
        fn dfft_cosine_has_peak_at_bin() {
            // x[n] = cos(2π k0 n / N) -> peaks at k0 and N-k0 with magnitude N/2 (unnormalized)
            let n = FFT_INPUT_SIZE;
            let k0 = 10usize; // ensure 2*k0 is not a multiple of N
            let mut x = [0.0f32; FFT_INPUT_SIZE];
            for i in 0..n {
                let theta = 2.0 * std::f32::consts::PI * (k0 as f32) * (i as f32) / (n as f32);
                x[i] = theta.cos();
            }
            let spectrum = dfft(x);

            // Find the maximum bin
            let (k_max, &v_max) = spectrum
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            // Fold symmetrical bin to baseband index
            let k_fold = if k_max > n / 2 { n - k_max } else { k_max };
            assert_eq!(
                k_fold, k0,
                "peak at unexpected bin: {} (folded {}), expected {}",
                k_max, k_fold, k0
            );

            let expected = (n as f32) / 2.0;
            let rel_err = (v_max - expected).abs() / expected.max(1.0);
            assert!(
                rel_err < 0.01,
                "peak magnitude off: got {}, expected ~{}",
                v_max,
                expected
            );
        }
    }
}
