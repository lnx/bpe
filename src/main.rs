use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::{BufReader, Read};

const VOCAB_SIZE: u32 = 1024;
const NUM_MERGES: u32 = VOCAB_SIZE - 256;

// training

fn train(ids: &[u32], num_merges: u32) -> HashMap<(u32, u32), u32> {
    println!("training: ids={}, num_merges={}", ids.len(), num_merges);
    let mut merges = HashMap::new();
    let mut ids = Vec::from(ids);
    for i in 0..num_merges {
        let stats = get_stats(&ids);
        if let Some((&pair, &_count)) = stats.iter().max_by_key(|&(_, v)| v) {
            // println!("merge:{}, pair:{:?}, count:{}", i, pair, _count);
            let idx = 256 + i;
            ids = merge(&ids, pair, idx);
            merges.insert(pair, idx);
        } else {
            break;
        }
    }
    merges
}

fn build_vocab(merges: &HashMap<(u32, u32), u32>) -> HashMap<u32, Vec<u8>> {
    let mut vocab = HashMap::new();
    for idx in 0..256_u32 {
        vocab.insert(idx, vec![idx as u8]);
    }
    let mut merges: Vec<_> = merges.iter().map(|(&p, &idx)| (idx, p.0, p.1)).collect();
    merges.sort_by_key(|&(idx, _, _)| idx);
    for &(idx, p0, p1) in &merges {
        let mut merged = vec![];
        merged.extend(&vocab[&p0]);
        merged.extend(&vocab[&p1]);
        vocab.insert(idx, merged);
    }
    vocab
}

fn get_stats(ids: &[u32]) -> HashMap<(u32, u32), u32> {
    let mut counts = HashMap::new();
    for pair in ids.windows(2) {
        *counts.entry((pair[0], pair[1])).or_default() += 1;
    }
    counts
}

fn merge(ids: &[u32], pair: (u32, u32), idx: u32) -> Vec<u32> {
    let mut new_ids = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if i < ids.len() - 1 && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            new_ids.push(idx);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }
    new_ids
}

// encoding

fn encode(merges: &HashMap<(u32, u32), u32>, text: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = text.as_bytes().iter().map(|&b| b.into()).collect();
    while ids.len() >= 2 {
        let pairs: Vec<(u32, u32)> = ids.windows(2).map(|p| (p[0], p[1])).collect();
        if let Some(&pair) = pairs
            .iter()
            .filter(|&k| merges.contains_key(k))
            .min_by_key(|&k| merges.get(k)) {
            ids = merge(&ids, pair, merges[&pair]);
        } else {
            break;
        }
    }
    ids
}

// decoding

fn decode(vocab: &HashMap<u32, Vec<u8>>, ids: &[u32]) -> String {
    let tokens: Vec<_> = ids.iter().flat_map(|idx| vocab[idx].clone()).collect();
    String::from_utf8_lossy(&tokens).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_stats() {
        let ids = vec![1, 2, 3, 1, 2];
        let stats = get_stats(&ids);
        assert_eq!(stats[&(1, 2)], 2);
        assert_eq!(stats[&(2, 3)], 1);
        assert_eq!(stats[&(3, 1)], 1);
    }

    #[test]
    fn test_merge() {
        let ids = vec![1, 2, 3, 1, 2];
        let new_ids = merge(&ids, (1, 2), 4);
        assert_eq!(new_ids, vec![4, 3, 4])
    }

    #[test]
    fn test_encode_decode() {
        let text = "The girl, unlike most people photographed for fashion magazines, was not beautiful.";
        let tokens: Vec<u32> = text.as_bytes().iter().map(|&b| b.into()).collect();
        let ids = tokens.clone();
        let merges = train(&ids, 512);
        let vocab = build_vocab(&merges);
        assert_eq!(decode(&vocab, &encode(&merges, text)), text);
    }
}

fn main() -> io::Result<()> {
    let f = File::open("a-man-like-him.txt")?;
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    // train
    let tokens: Vec<u32> = buffer.iter().map(|&b| b.into()).collect();
    let ids = tokens.clone();
    let merges = train(&ids, NUM_MERGES);
    let vocab = build_vocab(&merges);
    println!("merges:{}, vocab:{}", merges.len(), vocab.len());

    // encode & decode
    for text in vec![
        "hello world",
        "In the dusk, a thin mist hung in the air.",
        "The black-clad girl taunted him from the magazine lying open on the floor.",
        "李翊云：我觉得这里是两个问题，雷蒙德·卡佛是一个问题，《纽约客》是另一个问题。",
    ] {
        let ids = encode(&merges, text);
        let ratio = text.len() as f32 / ids.len() as f32;
        let decoded = decode(&vocab, &ids);
        println!("\n----------------------------------------");
        println!("text:    {}", text);
        println!("ids:     {:?}", ids);
        println!("ratio:   {:.2}", ratio);
        println!("decoded: {}", decoded);
    }

    Ok(())
}
