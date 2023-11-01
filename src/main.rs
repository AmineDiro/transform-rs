use anyhow::{anyhow, Result};

use std::io::{BufRead, BufReader};

use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};

use hf_hub;
use tokenizers::Tokenizer;
mod gpt;

fn load_tokenizer() -> Result<Tokenizer> {
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
    let tokenizer_path = api.get("tokenizer.json")?;
    let t = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;
    Ok(t)
}

fn load_dataset(dataset_path: &'static str, tokenizer: &Tokenizer) -> Result<Vec<Vec<u32>>> {
    let file = std::fs::File::open(dataset_path)?;
    let file = std::io::BufReader::new(file);
    let mut tokens = vec![];
    for line in file.lines() {
        let line = line.unwrap().replace("<|endoftext|>", "<s>");
        let line = tokenizer.encode(line, false).map_err(|e| anyhow!(e))?;
        tokens.push(line.get_ids().to_vec())
    }
    Ok(tokens)
}

fn main() -> Result<()> {
    let dataset_path = "./data/TinyStories-valid.txt";
    let tokenizer = load_tokenizer()?;
    let dataset = load_dataset(dataset_path, &tokenizer)?;
    println!(
        "Dataset loaded with : {} , First point length: {}",
        dataset.len(),
        dataset[0].len()
    );
    Ok(())
}
