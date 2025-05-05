use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use polars::prelude::*;

pub struct PolarsVectorstore {
    path: String,
    data: DataFrame,
}

pub struct SliceArgs {
    pub offset: i32,
    pub length: usize,
}

enum VstoreColumns {
    Embeddings,
}

impl Into<String> for VstoreColumns {
    fn into(self) -> String {
        match self {
            VstoreColumns::Embeddings => "embeddings".to_string(),
        }
    }
}

impl Into<PlSmallStr> for VstoreColumns {
    fn into(self) -> PlSmallStr {
        PlSmallStr::from_string(self.into())
    }
}

fn get_embeddings_dtype() -> DataType {
    DataType::List(Box::new(DataType::Float32))
}

fn get_empty_dataframe() -> DataFrame {
    let mut empty_df = DataFrame::empty();
    let empty_df_prepared = empty_df
        .with_column(Column::new_empty(
            VstoreColumns::Embeddings.into(),
            &get_embeddings_dtype(),
        ))
        .unwrap();
    empty_df_prepared.clone()
}

fn read_parquet(file_path: &str) -> DataFrame {
    let path = Path::new(file_path);
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| {
                    PolarsError::ComputeError(format!("Failed to create directory: {}", e).into())
                })
                .unwrap();
        }
    }

    let df = match std::fs::File::open(path) {
        Ok(mut f) => ParquetReader::new(&mut f).finish().unwrap(),
        Err(e) => {
            println!("Fail to load dataframe: {:?}", e);

            // Create an empty dataframe with proper structure
            let mut empty_df = get_empty_dataframe();
            let mut file = File::create(path).expect("Failed to create file");
            ParquetWriter::new(&mut file)
                .finish(&mut empty_df)
                .expect("Failed to write Parquet file");

            empty_df.clone()
        }
    };

    df
}

impl PolarsVectorstore {
    pub fn new(path: &str, empty: bool) -> Self {
        let lf = if empty {
            get_empty_dataframe()
        } else {
            read_parquet(path)
        };

        Self {
            path: path.to_string(),
            data: lf,
        }
    }

    pub fn reset(&mut self) {
        self.data.clear();
    }

    pub fn append(&mut self, vector: &Vec<f32>) -> Result<(), PolarsError> {
        self.append_many(&vec![vector.to_owned()])
    }

    pub fn append_many(&mut self, vectors: &Vec<Vec<f32>>) -> Result<(), PolarsError> {
        // Create Series with explicit List type to ensure consistent serialization

        let column = if vectors.is_empty() {
            Column::new_empty(VstoreColumns::Embeddings.into(), &get_embeddings_dtype())
        } else {
            let vec_series: Vec<Series> = vectors
                .iter()
                .map(|r| Series::new("".into(), r.as_slice()))
                .collect();
            let series = Series::new("".into(), vec_series);
            Column::new(VstoreColumns::Embeddings.into(), series)
        };

        let new_df = DataFrame::new(vec![column])?;
        self.data = self.data.vstack(&new_df)?;

        Ok(())
    }

    pub fn get_many(&self, slice: Option<SliceArgs>) -> Result<Vec<Vec<f32>>, PolarsError> {
        let col_name: String = VstoreColumns::Embeddings.into();

        let data = if let Some(args) = slice {
            &self.data.slice(args.offset.into(), args.length.into())
        } else {
            println!("DataFrame height: {}", self.data.height());
            &self.data
        };

        let column = data.column(&col_name)?;
        println!("Column data type: {:?}", column.dtype());
        println!("Column length: {}", column.len());

        // Convert column to Vec<Vec<f32>>
        let list_chunked = column.list()?;
        println!("List series length: {}", list_chunked.len());
        println!("List series dtype: {}", list_chunked.dtype());

        let result: Vec<Vec<f32>> = list_chunked
            .into_iter()
            .filter_map(|opt_series| {
                opt_series.map(|inner_series| {
                    inner_series
                        .f32()
                        .unwrap()
                        .into_iter()
                        .flatten()
                        .collect::<Vec<f32>>()
                })
            })
            .collect();

        println!("Final result length: {}", result.len());
        Ok(result)
    }

    pub fn get(&self, index: usize) -> Result<Vec<f32>, PolarsError> {
        match self
            .get_many(Some(SliceArgs {
                offset: index as i32,
                length: 1,
            }))?
            .get(0)
        {
            Some(val) => Ok(val.clone()),
            None => Err(PolarsError::NoData("Index not found".into())),
        }
    }

    pub fn reload(&mut self, force: bool) -> Result<(), PolarsError> {
        let new_df = read_parquet(&self.path);
        let nrows = new_df.height();

        if nrows == 0 && !force {
            return Err(PolarsError::NoData("Found a empty or invalid file".into()));
        }
        self.data = new_df;

        Ok(())
    }

    pub fn persist(&mut self) -> Result<(), PolarsError> {
        let path = Path::new(&self.path);

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    PolarsError::ComputeError(format!("Failed to create directory: {}", e).into())
                })?;
            }
        }

        println!("DataFrame schema before writing: {:?}", self.data.schema());
        println!("DataFrame height before writing: {}", self.data.height());

        {
            // Use a block to ensure file is closed properly
            let mut file = match File::create(path) {
                Ok(f) => f,
                Err(e) => {
                    return Err(PolarsError::ComputeError(
                        format!("Error creating file: {}", e).into(),
                    ));
                }
            };

            // Make sure to use consistent options when writing parquet
            let parquet_options = ParquetWriteOptions::default();
            match ParquetWriter::new(&mut file)
                .with_compression(parquet_options.compression)
                .with_data_page_size(parquet_options.data_page_size)
                .with_row_group_size(parquet_options.row_group_size)
                .with_statistics(parquet_options.statistics)
                .finish(&mut self.data)
            {
                Ok(_) => {}
                Err(e) => {
                    println!("Fail to write parquet: {:?}", e);
                    return Err(e);
                }
            };

            if let Err(e) = file.flush() {
                return Err(PolarsError::ComputeError(
                    format!("Error flushing file: {}", e).into(),
                ));
            }
        }

        if !path.exists() {
            return Err(PolarsError::ComputeError(
                format!("File was not created: {}", path.display()).into(),
            ));
        }

        let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        println!("File written successfully, size: {} bytes", file_size);

        Ok(())
    }

    pub fn get_count(&self) -> Result<usize, PolarsError> {
        let count = self.data.height();
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use crate::mpi_helper::get_global_vstore;
    use crate::utils::tests::*;

    use super::*;

    #[test]
    fn test_new_vectorstore() {
        let dir = get_vstore_dir();
        let mut store = get_global_vstore(dir.path(), true);
        let n = 3;
        sample_vstore(&mut store, n);

        assert_eq!(store.data.height(), n);

        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_append_vector() {
        let dir = get_vstore_dir();
        let mut store = get_global_vstore(dir.path(), true);
        let n = 1;
        let sample = sample_vstore(&mut store, n);

        let new_vector = generate_mock_embeddings();
        store.append(&new_vector).unwrap();

        let result = store.get_many(None).unwrap();
        println!("Count: {}", store.get_count().unwrap());
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], sample[0]);

        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_append_many_vectors() {
        let dir = get_vstore_dir();
        let mut store = get_global_vstore(dir.path(), true);
        let n = 3;
        sample_vstore(&mut store, n);

        let new_vectors = vec![generate_mock_embeddings(), generate_mock_embeddings()];
        store.append_many(&new_vectors).unwrap();

        let result = store
            .get_many(Some(SliceArgs {
                offset: n as i32,
                length: new_vectors.len(),
            }))
            .unwrap();
        assert_eq!(result.len(), new_vectors.len());
        assert_eq!(result[0], new_vectors[0]);
        assert_eq!(result[1], new_vectors[1]);

        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_read_slice() {
        let dir = get_vstore_dir();
        let mut store = get_global_vstore(dir.path(), true);
        let n = 3;
        let sample = sample_vstore(&mut store, n);

        // Read a slice from the middle
        let result = store
            .get_many(Some(SliceArgs {
                offset: 1,
                length: 1,
            }))
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], sample[1]);

        // Read multiple rows
        let result = store.get_many(None).unwrap();
        assert_eq!(result.len(), n);
        assert_eq!(result[0], sample[0]);
        assert_eq!(result[1], sample[1]);
        assert_eq!(result[2], sample[2]);

        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_persist_and_reload() {
        let dir = get_vstore_dir();
        let mut store = get_global_vstore(dir.path(), true);
        let n = 3;
        sample_vstore(&mut store, n);

        store.append(&generate_mock_embeddings()).unwrap();
        store.persist().unwrap();

        let new_store = get_global_vstore(dir.path(), false);
        let result = new_store.get_many(None).unwrap();
        assert_eq!(result.len(), n + 1);

        dir.close().unwrap_or_default()
    }

    #[test]
    fn test_empty_file_reload() {
        let dir = get_vstore_dir();
        let mut store = get_global_vstore(&dir.path(), true);

        let result = store.reload(false);
        assert!(result.is_err());

        let result = store.reload(true);
        assert!(result.is_ok());

        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_large_dataset() {
        let dir = get_vstore_dir();
        let mut store = get_global_vstore(dir.path(), true);
        let n = 1000;
        sample_vstore(&mut store, n);

        let result = store
            .get_many(Some(SliceArgs {
                offset: 0,
                length: 100,
            }))
            .unwrap();
        assert_eq!(result.len(), 100);

        let result = store
            .get_many(Some(SliceArgs {
                offset: 500,
                length: 200,
            }))
            .unwrap();
        assert_eq!(result.len(), 200);

        let result = store.get_many(None).unwrap();
        assert_eq!(result.len(), 1000);

        dir.close().unwrap_or_default();
    }
}
