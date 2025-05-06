use ndarray::{Array2, Axis, Ix2};
use ort::{Error, Result, execution_providers::CUDAExecutionProvider, session::Session};
use std::path::Path;
use tokenizers::Tokenizer;

use super::LlmService;

fn build_model_and_tokenizer() -> Result<(Session, Tokenizer), ort::Error> {
    let models_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(".volumes/models/nomic_embed_text_onnx");

    let model = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(&models_dir.join("model.onnx"))?;
    let tokenizer = Tokenizer::from_file(&models_dir.join("tokenizer.json")).unwrap();

    Ok((model, tokenizer))
}

pub struct HfService {
    model: Session,
    tokenizer: Tokenizer,
}

impl LlmService for HfService {
    type Error = ort::Error;

    fn default() -> Result<Self> {
        tracing_subscriber::fmt::init();

        ort::init()
            .with_name("nomic-embed-text")
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .commit()?;

        let (model, tokenizer) = build_model_and_tokenizer()?;
        Ok(Self { model, tokenizer })
    }

    // fn count_tokens(&self, text: String) -> Result<usize> {
    //     match self.tokenizer.encode(text, false) {
    //         Ok(t) => Ok(t.len()),
    //         Err(e) => Err(Error::new(e.to_string()))

    //     }
    // }

    fn get_embeddings(&self, texts: &Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if texts.iter().any(|v| v.is_empty()) {
            return Err(Error::new("Invalid inputs: has empty values"));
        }

        // Encode our input strings. `encode_batch` will pad each input to be the same length.
        let encodings = self
            .tokenizer
            .encode_batch(texts.clone(), false)
            .map_err(|e| Error::new(e.to_string()))?;
        // Get the padded length of each encoding.
        let padded_token_length = encodings[0].len();

        let has_same_shape = encodings.iter().all(|v| v.len() == padded_token_length);
        if !has_same_shape {
            let mut shape_example: Vec<&str> = Vec::new();
            let mut shapes: Vec<usize> = Vec::new();
            encodings.iter().enumerate().for_each(|(i, v)| {
                let v_len = v.len();
                if !shapes.contains(&v_len) {
                    shapes.push(v_len);
                    shape_example.push(&texts[i]);
                }
            });
            println!(
                "Shape inconsistent: {:?}",
                shapes
            );
        }

        // Get our token IDs & mask as a flattened array.
        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();

        // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
        // TODO: make exception log better
        let a_ids = Array2::from_shape_vec([texts.len(), padded_token_length], ids).unwrap();
        let a_mask = Array2::from_shape_vec([texts.len(), padded_token_length], mask).unwrap();

        // Run the model.
        let outputs = self.model.run(ort::inputs![a_ids, a_mask]?)?;

        // Extract our embeddings tensor and convert it to a strongly-typed 2-dimensional array.
        let embeddings = outputs[1]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix2>()
            .unwrap();
        let result: Vec<Vec<f32>> = embeddings
            .axis_iter(Axis(0))
            .map(|row| row.to_vec())
            .collect();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tests::DIMENSION;
    use std::path::Path;
    use std::sync::Once;

    // Initialize tracing only once for all tests
    static TRACING_INIT: Once = Once::new();

    // Helper to check if model files exist
    fn model_files_exist() -> bool {
        let models_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(".volumes/models/nomic_embed_text_onnx");
        models_path.exists()
            && models_path.join("model.onnx").exists()
            && models_path.join("tokenizer.json").exists()
    }

    // Helper to create a test-specific HfService that handles environments without CUDA
    fn create_test_service() -> Result<HfService> {
        // Initialize tracing only once
        TRACING_INIT.call_once(|| {
            // Use a no-op implementation for tests to avoid console output
            let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        });

        // Initialize ONNX Runtime in a test-friendly way
        let runtime_result = ort::init()
            .with_name("test-nomic-embed-text")
            .with_global_thread_pool(
                ort::environment::GlobalThreadPoolOptions::default()
                    .with_inter_threads(1)
                    .unwrap(),
            );

        runtime_result.commit().unwrap();

        // Create service with the model and tokenizer
        let (model, tokenizer) = build_model_and_tokenizer()?;
        Ok(HfService { model, tokenizer })
    }

    #[test]
    fn test_get_embeddings_single_text() -> Result<()> {
        if !model_files_exist() {
            println!("Skipping test: model files not found");
            return Ok(());
        }

        // Create service using our test helper
        let service = create_test_service()?;

        // Use a short text to minimize computation
        let texts = vec!["Hello".to_string()];

        // Generate embeddings
        let result = service.get_embeddings(&texts)?;

        // Basic assertions
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), DIMENSION); // Embedding dimension should be equal to DIMENSION

        // Check that embedding isn't all zeros or all the same value
        let first_val = result[0][0];
        let all_same = result[0]
            .iter()
            .all(|&val| (val - first_val).abs() < f32::EPSILON);
        assert!(!all_same, "All embedding values should not be identical");

        // Check if embedding contains non-zero values
        let has_nonzeros = result[0].iter().any(|&val| val.abs() > f32::EPSILON);
        assert!(has_nonzeros, "Embedding should contain non-zero values");

        Ok(())
    }

    #[test]
    fn test_get_embeddings_multiple_texts() -> Result<()> {
        if !model_files_exist() {
            println!("Skipping test: model files not found");
            return Ok(());
        }

        // Create service using our test helper
        let service = create_test_service()?;

        // Use short texts to minimize computation
        let texts = vec!["Hi".to_string(), "Hello".to_string()];

        // Generate embeddings
        let result = service.get_embeddings(&texts)?;

        // Basic assertions
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), DIMENSION);
        assert_eq!(result[1].len(), DIMENSION);

        // Check that embeddings are different (since texts are different)
        let mut different = false;
        for i in 0..DIMENSION {
            if (result[0][i] - result[1][i]).abs() > 1e-5 {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "Embeddings for different texts should be different"
        );

        Ok(())
    }

    #[test]
    fn test_get_embeddings_empty_text() -> Result<()> {
        if !model_files_exist() {
            println!("Skipping test: model files not found");
            return Ok(());
        }

        // Create service using our test helper
        let service = create_test_service()?;

        // Empty text
        let texts: Vec<String> = vec!["".to_string()];

        // Generate embeddings
        let result = service.get_embeddings(&texts);

        // Basic assertions
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().message(),
            "Invalid inputs: has empty values"
        );

        Ok(())
    }

    #[test]
    fn test_get_embeddings_empty_vec() -> Result<()> {
        if !model_files_exist() {
            println!("Skipping test: model files not found");
            return Ok(());
        }

        // Create service using our test helper
        let service = create_test_service()?;

        // Empty text
        let texts: Vec<String> = vec![];

        // Generate embeddings
        let result = service.get_embeddings(&texts)?;

        // Basic assertions
        assert!(result.is_empty());

        Ok(())
    }

    #[test]
    fn test_embedding_consistency() -> Result<()> {
        if !model_files_exist() {
            println!("Skipping test: model files not found");
            return Ok(());
        }

        // Create service using our test helper
        let service = create_test_service()?;

        // Generate embeddings for the same text twice
        let text = "consistency test".to_string();
        let texts1 = vec![text.clone()];
        let texts2 = vec![text];

        let result1 = service.get_embeddings(&texts1)?;
        let result2 = service.get_embeddings(&texts2)?;

        // Embeddings for the same text should be identical
        for i in 0..DIMENSION {
            assert!(
                (result1[0][i] - result2[0][i]).abs() < f32::EPSILON,
                "Embeddings should be consistent for the same input"
            );
        }

        Ok(())
    }
}
