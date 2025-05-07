// src/metrics.rs
use std::collections::HashMap;

type Serie = Vec<f32>;

// Helper function to directly check if vectors are identical
fn vectors_are_identical(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > 1e-10 {
            return false;
        }
    }

    true
}

pub fn mean_reciprocal_rank(
    queries: &Vec<Vec<f32>>,
    results: &Vec<Vec<Vec<f32>>>,
    ids_map: Option<&HashMap<usize, usize>>,
) -> f32 {
    if queries.is_empty() || results.is_empty() || queries.len() != results.len() {
        return 0.0;
    }

    let mut sum_reciprocal_ranks = 0.0;
    let total_queries = queries.len() as f32;

    for (query_idx, query) in queries.iter().enumerate() {
        let query_results = &results[query_idx];
        let mut rank = 0;

        for (i, result) in query_results.iter().enumerate() {
            let _result_idx = match ids_map {
                Some(map) => map.get(&i).copied().unwrap_or(i),
                None => i,
            };

            // Use direct vector comparison instead of cosine distance
            if vectors_are_identical(query, result) {
                rank = i + 1;
                break;
            }
        }

        if rank == 0 {
            rank = query_results.len() + 1;
        }

        sum_reciprocal_ranks += 1.0 / (rank as f32);
    }

    sum_reciprocal_ranks / total_queries
}

pub fn recall_at_k(
    queries: &Vec<Vec<f32>>,
    results: &Vec<Vec<Vec<f32>>>,
    k: usize,
    ids_map: Option<&HashMap<usize, usize>>,
) -> f32 {
    if queries.is_empty() || results.is_empty() || queries.len() != results.len() {
        return 0.0;
    }

    let mut sum_recall = 0.0;
    let total_queries = queries.len() as f32;

    for (query_idx, query) in queries.iter().enumerate() {
        let query_results = &results[query_idx];
        let actual_k = std::cmp::min(k, query_results.len());

        let mut found_in_top_k = false;

        for i in 0..actual_k {
            let _result_idx = match ids_map {
                Some(map) => map.get(&i).copied().unwrap_or(i),
                None => i,
            };

            // Use direct vector comparison instead of cosine distance
            if vectors_are_identical(query, &query_results[i]) {
                found_in_top_k = true;
                break;
            }
        }

        sum_recall += if found_in_top_k { 1.0 } else { 0.0 };
    }

    sum_recall / total_queries
}

pub fn top_k_overlap(
    sequential_results: &Vec<Vec<Vec<f32>>>,
    parallel_results: &Vec<Vec<Vec<f32>>>,
    k: usize,
) -> f32 {
    if sequential_results.is_empty()
        || parallel_results.is_empty()
        || sequential_results.len() != parallel_results.len()
    {
        return 0.0;
    }

    let mut sum_overlap = 0.0;
    let total_queries = sequential_results.len() as f32;

    for (query_idx, seq_results) in sequential_results.iter().enumerate() {
        let par_results = &parallel_results[query_idx];
        let actual_k = std::cmp::min(k, std::cmp::min(seq_results.len(), par_results.len()));

        if actual_k == 0 {
            continue;
        }

        let mut overlap_count = 0;

        for i in 0..actual_k {
            let seq_item = &seq_results[i];

            for j in 0..actual_k {
                let par_item = &par_results[j];

                // Use direct vector comparison instead of cosine distance
                if vectors_are_identical(seq_item, par_item) {
                    overlap_count += 1;
                    break;
                }
            }
        }

        sum_overlap += overlap_count as f32 / actual_k as f32;
    }

    sum_overlap / total_queries
}

pub fn cosine_distance(a: &Serie, b: &Serie) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }

    // Early return if vectors are identical
    if vectors_are_identical(a, b) {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(a_i, b_i)| a_i * b_i).sum();
    let magnitude_a: f32 = a.iter().map(|a_i| a_i * a_i).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|b_i| b_i * b_i).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 1.0;
    }

    let cosine_similarity = dot_product / (magnitude_a * magnitude_b);
    let cosine_similarity = cosine_similarity.max(-1.0).min(1.0);

    1.0 - cosine_similarity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tests::*;

    #[test]
    fn test_mrr_perfect_match() {
        let queries = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let mut results = Vec::new();
        for query in &queries {
            results.push(vec![query.clone()]);
        }

        let mrr = mean_reciprocal_rank(&queries, &results, None);
        assert!(
            (mrr - 1.0).abs() < 1e-5,
            "Expected MRR of 1.0 for perfect matches"
        );
    }

    #[test]
    fn test_mrr_mixed_ranks() {
        let embeddings = generate_many_mock_embeddings(3);
        let queries = embeddings.clone();

        let mut results = Vec::new();

        let mut result0 = Vec::new();
        result0.push(embeddings[0].clone());
        result0.push(generate_mock_embeddings());
        result0.push(generate_mock_embeddings());
        results.push(result0);

        let mut result1 = Vec::new();
        result1.push(generate_mock_embeddings());
        result1.push(embeddings[1].clone());
        result1.push(generate_mock_embeddings());
        results.push(result1);

        let mut result2 = Vec::new();
        result2.push(generate_mock_embeddings());
        result2.push(generate_mock_embeddings());
        result2.push(embeddings[2].clone());
        results.push(result2);

        let mrr = mean_reciprocal_rank(&queries, &results, None);
        assert!(
            (mrr - 0.611).abs() < 0.01,
            "Expected MRR around 0.611 for mixed ranks"
        );
    }

    #[test]
    fn test_recall_at_k() {
        let embeddings = generate_many_mock_embeddings(4);
        let queries = embeddings.clone();

        let mut results = Vec::new();

        let mut result0 = Vec::new();
        result0.push(embeddings[0].clone());
        result0.extend(generate_many_mock_embeddings(5));
        results.push(result0);

        let mut result1 = Vec::new();
        result1.push(generate_mock_embeddings());
        result1.push(embeddings[1].clone());
        result1.extend(generate_many_mock_embeddings(4));
        results.push(result1);

        let mut result2 = Vec::new();
        result2.push(generate_mock_embeddings());
        result2.push(generate_mock_embeddings());
        result2.push(embeddings[2].clone());
        result2.extend(generate_many_mock_embeddings(3));
        results.push(result2);

        let mut result3 = Vec::new();
        result3.push(generate_mock_embeddings());
        result3.push(generate_mock_embeddings());
        result3.push(generate_mock_embeddings());
        result3.push(generate_mock_embeddings());
        result3.push(embeddings[3].clone());
        result3.extend(generate_many_mock_embeddings(1));
        results.push(result3);

        let recall1 = recall_at_k(&queries, &results, 1, None);
        assert!((recall1 - 0.25).abs() < 1e-5, "Expected Recall@1 of 0.25");

        let recall2 = recall_at_k(&queries, &results, 2, None);
        assert!((recall2 - 0.5).abs() < 1e-5, "Expected Recall@2 of 0.5");

        let recall3 = recall_at_k(&queries, &results, 3, None);
        assert!((recall3 - 0.75).abs() < 1e-5, "Expected Recall@3 of 0.75");

        let recall5 = recall_at_k(&queries, &results, 5, None);
        assert!((recall5 - 1.0).abs() < 1e-5, "Expected Recall@5 of 1.0");
    }

    #[test]
    fn test_top_k_overlap() {
        let embeddings = generate_many_mock_embeddings(10);

        let mut sequential_results = Vec::new();
        let mut parallel_results = Vec::new();

        let seq_results1 = vec![
            embeddings[0].clone(),
            embeddings[1].clone(),
            embeddings[2].clone(),
            embeddings[3].clone(),
            embeddings[4].clone(),
        ];
        let par_results1 = vec![
            embeddings[0].clone(),
            embeddings[1].clone(),
            embeddings[5].clone(),
            embeddings[3].clone(),
            embeddings[6].clone(),
        ];
        sequential_results.push(seq_results1);
        parallel_results.push(par_results1);

        let seq_results2 = vec![
            embeddings[0].clone(),
            embeddings[1].clone(),
            embeddings[2].clone(),
            embeddings[3].clone(),
            embeddings[4].clone(),
        ];
        let par_results2 = vec![
            embeddings[0].clone(),
            embeddings[1].clone(),
            embeddings[2].clone(),
            embeddings[3].clone(),
            embeddings[4].clone(),
        ];
        sequential_results.push(seq_results2);
        parallel_results.push(par_results2);

        let overlap = top_k_overlap(&sequential_results, &parallel_results, 5);
        assert!(
            (overlap - 0.8).abs() < 0.01,
            "Expected Top-5 Overlap of 0.8"
        );
    }
}
