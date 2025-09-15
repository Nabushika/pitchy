use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use cpal::{FromSample, SizedSample};

use std::sync::mpsc;

use tuner_core::{
    correct_chunk_size, fast_dfft, freq_to_note, max_freq, note_to_name, RawChunkedPacket,
};

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
