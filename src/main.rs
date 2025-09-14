use cpal::{FromSample, SizedSample};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;

use std::collections::VecDeque;
use std::sync::mpsc;

const FFT_INPUT_SIZE: usize = 8192; //(48000 / 20 as usize).next_power_of_two(); // 20Hz lowest freq, rounded up to a power of two
const FFT_OUTPUT_SIZE: usize = FFT_INPUT_SIZE / 2;
type RawChunkedPacket = Vec<f32>;
type CorrectedChunkedPacket = [f32; FFT_INPUT_SIZE];
type DFFTPacket = [f32; FFT_OUTPUT_SIZE]; // cut off Nyquist

trait SizedArray {
    const LEN: usize;
}

impl<const N: usize, T> SizedArray for [T; N] {
    const LEN: usize = N;
}

fn main() {
    use std::io::Write;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    let running = Arc::new(AtomicBool::new(true));
    let (handle, raw_chunk_rx) = listen();
    let corrected_chunk_rx = correct_chunk_size(raw_chunk_rx);

    handle.0.play().unwrap();

    ctrlc::set_handler({
        let running = running.clone();
        move || {
            running.store(false, Ordering::SeqCst);
        }
    })
    .expect("Error setting Ctrl-C handler");

    println!("Press CTRL+C to exit...");
    let mut stdout = std::io::stdout().lock();

    while running.load(Ordering::SeqCst) {
        let Ok(chunk) = corrected_chunk_rx.recv() else {
            eprintln!("Failed to receive chunk, stream ended, exiting");
            break;
        };
        let spectrum = fast_dfft(chunk);
        let freq = max_freq(&spectrum, 48000.0 / 2.0); // hack for half-sized fft output
        let (note, detune) = freq_to_note(freq);
        let note_name = note_to_name(note);
        let freq_rounded = freq.round();

        let msg = format!(
            "Frequency: {} Hz ({} +- {} cents)\r",
            freq_rounded, note_name, detune
        );

        stdout.write_all(format!("{msg: <70}").as_bytes()).unwrap();
        stdout.flush().unwrap();
    }
}

#[allow(dead_code)]
fn dfft(chunk: CorrectedChunkedPacket) -> DFFTPacket {
    let mut output = [0.0; FFT_OUTPUT_SIZE];
    const N: usize = FFT_OUTPUT_SIZE;
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

#[allow(clippy::assertions_on_constants)] // keep the assertion for safety
fn fast_dfft(chunk: CorrectedChunkedPacket) -> DFFTPacket {
    use rustfft::{num_complex::Complex, Fft};
    use std::sync::{Arc, LazyLock};
    debug_assert!(CorrectedChunkedPacket::LEN.is_power_of_two());
    // screw it, let's use a library for now
    static FFT_PLAN: LazyLock<Arc<dyn Fft<f32>>> = LazyLock::new(|| {
        let mut planner = rustfft::FftPlanner::<f32>::new();
        planner.plan_fft_forward(CorrectedChunkedPacket::LEN)
    });
    let mut buf = chunk.map(|x| Complex::new(x, 0.0));
    FFT_PLAN.process(&mut buf);
    // Can't do this because we can't build an array from an iterator :c
    //buf.into_iter().map(|x| x.norm()).collect()
    // Use intermediate allocation instead
    let whole_buf = buf.map(|x| x.norm());
    assert!(DFFTPacket::LEN * 2 == CorrectedChunkedPacket::LEN);
    // SAFETY: whole_buf is a slice of length DFFTPacket::LEN * 2, so it can be cast to a slice of length DFFTPacket::LEN
    let [result, _] = unsafe { *(whole_buf.as_ptr() as *const [DFFTPacket; 2]) };
    result
}

// Compute the dominant frequency (in Hz) from a magnitude spectrum.
// Returns 0.0 if all values are non-finite (should not happen for normal inputs).
fn max_freq(spectrum: &[f32], sample_rate: f32) -> f32 {
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
const A4_FREQ: f32 = 440.0;
const A4_NOTE_NUM: i32 = 69;
const A4_OCTAVE: i32 = 4;
// Returns MIDI note number, + fractional detune (in cents)
fn freq_to_note(freq: f32) -> (i32, f32) {
    let a4_diff = 12.0 * (freq / A4_FREQ).log2();
    let note = a4_diff.round() as i32;
    (A4_NOTE_NUM + note, (a4_diff - note as f32) * 100.0)
}
// Note number to human-readable note name
fn note_to_name(note: i32) -> String {
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    // 69 => A4
    // 71 => B4
    // 72 => C5 because the octave starts on C
    let note_diff_c4 = note - A4_NOTE_NUM + 9;
    let note_idx = note_diff_c4.rem_euclid(12) as usize;
    let octave = (note_diff_c4.div_euclid(12)) + A4_OCTAVE;
    format!("{}{}", note_names[note_idx], octave)
}

fn correct_chunk_size(
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

pub struct Handle(pub Stream);

pub fn beep() -> Handle {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("failed to find a default output device");
    let config = device.default_output_config().unwrap();

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()),
        // not all supported sample formats are included in this example
        _ => panic!("Unsupported sample format!"),
    };

    Handle(stream)
}

pub fn listen() -> (Handle, mpsc::Receiver<RawChunkedPacket>) {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("failed to find a default input device");
    let config = device
        .default_input_config()
        .expect("failed to find a default input config");
    println!("{:?}", config);

    let (stream, raw_chunk_rx) = match config.sample_format() {
        cpal::SampleFormat::F32 => listen_run::<f32>(&device, &config.into()),
        cpal::SampleFormat::I16 => listen_run::<i16>(&device, &config.into()),
        cpal::SampleFormat::U16 => listen_run::<u16>(&device, &config.into()),
        // not all supported sample formats are included in this example
        _ => panic!("Unsupported sample format!"),
    };
    (Handle(stream), raw_chunk_rx)
}

fn listen_run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
) -> (Stream, mpsc::Receiver<RawChunkedPacket>)
where
    T: SizedSample + FromSample<f32> + Into<f32>,
{
    let (tx, rx) = mpsc::channel();

    (
        device
            .build_input_stream(
                config,
                move |xs: &[T], _| {
                    let chunk: RawChunkedPacket = xs.iter().map(|x| (*x).into()).collect();
                    tx.send(chunk).unwrap();
                },
                |err| println!("an error occurred on stream: {}", err),
                None,
            )
            .expect("failed to build input stream"),
        rx,
    )
}

fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Stream
where
    T: SizedSample + FromSample<f32>,
{
    let sample_rate = config.sample_rate.0 as f32;
    let channels = config.channels as usize;

    // Produce a sinusoid of maximum amplitude.
    let mut sample_clock = 0f32;
    let mut next_value = move || {
        sample_clock = (sample_clock + 1.0) % sample_rate;
        (sample_clock * 440.0 * 2.0 * std::f32::consts::PI / sample_rate).sin()
    };

    let err_fn = |err| println!("an error occurred on stream: {}", err);

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _| write_data(data, channels, &mut next_value),
            err_fn,
            None,
        )
        .unwrap();
    stream.play().unwrap();
    stream
}

fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> f32)
where
    T: SizedSample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let value: T = T::from_sample(next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}

#[cfg(test)]
mod tests {
    mod notes {
        use super::super::{freq_to_note, note_to_name};

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
            // Testing simple octaves
            let simple_octave_tests = [
                (440.0, (69, 0.0)),
                (880.0, (69 + 12, 0.0)),
                (220.0, (69 - 12, 0.0)),
            ];
            // Testing known semitones, no detune
            let semitone_tests = [
                (4186.0, (108, 0.0)), // C8
                (1864.7, (94, 0.0)),  // A#6
            ];
            // Test detune
            let detune_tests = [(441.0, (69, 4.0)), (445.0, (69, 20.0))];

            let all_tests = simple_octave_tests
                .into_iter()
                .chain(semitone_tests.into_iter())
                .chain(detune_tests.into_iter());

            for (freq, (exp_note, exp_detune)) in all_tests {
                println!("Freq {freq} should be note {exp_note} with detune {exp_detune}");
                let (note, detune) = freq_to_note(freq);
                println!("Calculated: note {note}, detune {detune}");
                assert_eq!(note, exp_note);
                assert!((detune - exp_detune).abs() < 0.5); // Allow 0.5 cents of error
            }
        }
    }
    mod fft {
        use super::super::{dfft, FFT_INPUT_SIZE};

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
