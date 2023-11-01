use candle_core::{Device, Tensor};
use rand::{seq::SliceRandom, thread_rng};

pub enum Split {
    Train,
    Test,
}
pub struct Dataset<'a> {
    train_tokens: &'a mut [u32],
    test_tokens: &'a mut [u32],
}

impl<'a> Dataset<'a> {
    pub fn new(tokens: &'a mut [u32], split_ratio: f32) -> Self {
        let train_idx = (tokens.len() as f32 * split_ratio) as usize;
        let (train_tokens, test_tokens) = tokens.split_at_mut(train_idx);

        Self {
            train_tokens,
            test_tokens,
        }
    }

    pub fn train_iter(&mut self, block_size: usize, device: Device) -> DatasetIter {
        DatasetIter::new(self, Split::Train, block_size, device)
    }

    pub fn train_len(&self) -> usize {
        self.train_tokens.len()
    }
    pub fn test_len(&self) -> usize {
        self.test_tokens.len()
    }
}

struct DatasetIter<'a> {
    tokens: &'a mut [u32],
    block_size: usize,
    device: Device,
}

impl<'a> DatasetIter<'a> {
    pub fn new(dataset: &'a mut Dataset, split: Split, block_size: usize, device: Device) -> Self {
        let tokens = match split {
            Split::Train => {
                let tokens = &mut dataset.train_tokens;
                tokens.shuffle(&mut thread_rng());
                tokens
            }
            Split::Test => {
                let tokens = &mut dataset.test_tokens;
                tokens.shuffle(&mut thread_rng());
                tokens
            }
        };

        Self {
            tokens: *tokens,
            block_size,
            device,
        }
    }
}

impl<'a> Iterator for DatasetIter<'a> {
    // Returns X, y
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn read_data() {
    let input_path = "./data/input.txt";
    let tokenizer = Tokenizer::from_file("./model/tokenizer.json").unwrap();

    // Tokenize dataset
    let input_file = fs::File::open(&input_path).unwrap();
    let input = BufReader::new(input_file);
    let mut tokens: Vec<u32> = input
        .lines()
        .map(|l| {
            let line = l.unwrap();
            let line_encoding = tokenizer.encode(line, false).unwrap();
            line_encoding.get_ids().to_vec()
        })
        .flatten()
        .collect();

    let dataset = Dataset::new(&mut tokens, 0.9);
    assert_eq!(dataset.test_len() + dataset.train_len(), tokens.len())
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_dataset_iter() {
        let mut tokens: Vec<u32> = vec![1, 2, 3, 4];
        let mut dataset = Dataset::new(&mut tokens, 0.5);
        let device = Device::Cpu;
        let train_dataset: Vec<(Tensor, Tensor)> = dataset.train_iter(1, device).collect();
        assert_eq!(train_dataset, [1, 2])
    }
}
