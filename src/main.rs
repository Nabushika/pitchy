use cpal::{FromSample, SizedSample};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;

use std::collections::VecDeque;
use std::sync::mpsc;

const FFT_CHUNK_SIZE: usize = (48000 / 20 as usize).next_power_of_two(); // 20Hz lowest freq, rounded up to a power of two
type RawChunkedPacket = Vec<f32>;
type CorrectedChunkedPacket = [f32; FFT_CHUNK_SIZE];
type DFFTPacket = [f32; FFT_CHUNK_SIZE];

trait SizedArray {
    const LEN: usize;
}

impl<const N: usize, T> SizedArray for [T; N] {
    const LEN: usize = N;
}

fn main() {
    let (handle, raw_chunk_rx) = listen();
    let _corrected_chunk_rx = correct_chunk_size(raw_chunk_rx);

    handle.0.play().unwrap();

    std::thread::sleep(std::time::Duration::from_secs(1));
}

fn dfft(chunk: CorrectedChunkedPacket) -> DFFTPacket {
    let mut output = [0.0; FFT_CHUNK_SIZE];
    const N: usize = FFT_CHUNK_SIZE;
    for k in 0..N {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for n in 0..N {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / N as f32;
            let (s, c) = angle.sin_cos();
            sum_re += chunk[n] * c;
            sum_im += chunk[n] * s;
        }
        output[k] = (sum_re * sum_re + sum_im * sum_im).sqrt();
    }
    output
}

fn fast_dfft(chunk: CorrectedChunkedPacket) -> DFFTPacket {
    debug_assert!(CorrectedChunkedPacket::LEN.is_power_of_two());
    debug_assert!(DFFTPacket::LEN == CorrectedChunkedPacket::LEN);
    todo!()
}


fn correct_chunk_size(
    rx: mpsc::Receiver<RawChunkedPacket>,
) -> mpsc::Receiver<CorrectedChunkedPacket> {
    let (tx, ret_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buffer = VecDeque::new();
        for chunk in rx {
            buffer.extend(chunk);
            while buffer.len() >= FFT_CHUNK_SIZE {
                let sized_chunk: CorrectedChunkedPacket = buffer
                    .drain(0..FFT_CHUNK_SIZE)
                    .into_iter()
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
        (sample_clock * 440.0 * 2.0 * 3.141592 / sample_rate).sin()
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
    use super::{dfft, FFT_CHUNK_SIZE};

    #[test]
    fn dfft_delta_is_flat_one() {
        // x[n] = delta[n] -> X[k] = 1 for all k (unnormalized DFT)
        let mut x = [0.0f32; FFT_CHUNK_SIZE];
        x[0] = 1.0;
        let spectrum = dfft(x);
        for &v in spectrum.iter() {
            assert!((v - 1.0).abs() < 1e-4, "expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn dfft_cosine_has_peak_at_bin() {
        // x[n] = cos(2π k0 n / N) -> peaks at k0 and N-k0 with magnitude N/2 (unnormalized)
        let n = FFT_CHUNK_SIZE;
        let k0 = 10usize; // ensure 2*k0 is not a multiple of N
        let mut x = [0.0f32; FFT_CHUNK_SIZE];
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
        assert_eq!(k_fold, k0, "peak at unexpected bin: {} (folded {}), expected {}", k_max, k_fold, k0);

        let expected = (n as f32) / 2.0;
        let rel_err = (v_max - expected).abs() / expected.max(1.0);
        assert!(rel_err < 0.01, "peak magnitude off: got {}, expected ~{}", v_max, expected);
    }
}
