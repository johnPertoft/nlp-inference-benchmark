use rust_bert::{bert::{BertConfig, BertForSentenceEmbeddings}};
use tch::{nn, Device, Tensor, no_grad, Kind};

fn main() {
    let device = Device::Cuda(0);

    let config = BertConfig::default();
    let vs = nn::VarStore::new(device);
    let model = BertForSentenceEmbeddings::new(&vs.root(), &config);

    let sequence_length = 512;

    // Warmup, necessary?
    for _ in 0..10 {
        let batch_size = 32;
        let input_ids = Tensor::ones(&[batch_size, sequence_length], (Kind::Int64, device));
        let _ = no_grad(|| {
            model.forward_t(
                Some(&input_ids),
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )
        });
    }

    let batch_sizes = vec![1]; //vec![1, 2, 4, 8, 16, 32, 64, 128, 256];
    for batch_size in batch_sizes {
        let input_ids = Tensor::ones(&[batch_size, sequence_length], (Kind::Int64, device));
        let n_measurements = 1000;
        let timings = (0..n_measurements).map(|_| {
            let t_start = std::time::Instant::now();
            let _ = no_grad(|| {
                model.forward_t(
                    Some(&input_ids),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
            });
            let t_end = std::time::Instant::now();
            
            let latency = t_end - t_start;
            let latency = latency.as_micros();
            let latency = latency as f32;
            let latency = latency / 1000000.0;
            latency
        });

        let avg_duration: f32 = timings.sum::<f32>() / (n_measurements as f32);
        println!("Batch size {batch_size} Avg latency {avg_duration}");
    }
}