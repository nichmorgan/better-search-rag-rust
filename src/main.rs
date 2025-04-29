mod llm;

use anyhow::Result;
use mpi::{
    collective::SystemOperation,
    traits::{Communicator, CommunicatorCollectives},
};

#[tokio::main]
async fn main() -> Result<()> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    println!("Process {} of {} initialized", rank, size);

    let llm_service = llm::LlmService::default();

    let text = format!("This is a test message from rank {}", rank);
    let embeddings = llm_service.get_embeddings(vec![&text]).await?.get(0).unwrap();
    println!(
        "Rank {}: Generated embedding with {} dimensions",
        rank,
        embeddings.len()
    );

    let mut sum_buffer = vec![embeddings.len() as i32];
    world.all_reduce_into(
        &[embeddings.len() as i32],
        &mut sum_buffer,
        &SystemOperation::sum(),
    );
    println!("Rank {}: Total embeddings dimensions across all ranks: {}", rank, sum_buffer[0]);

    drop(world);
    Ok(())
}
