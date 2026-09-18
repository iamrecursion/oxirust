//! Pipeline builder APIs for constructing complex data processing workflows

#![allow(dead_code)]
#![allow(missing_docs)]

use super::*;
use crate::error::Result;
use std::marker::PhantomData;

/// Fluent builder for constructing pipelines
pub struct PipelineBuilder<I, O> {
    stages: Vec<Box<dyn PipelineStage>>,
    config: PipelineConfig,
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

impl<I, O> Default for PipelineBuilder<I, O>
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<I, O> PipelineBuilder<I, O>
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: PipelineConfig::default(),
            _input: PhantomData,
            _output: PhantomData,
        }
    }

    /// Set parallel execution
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.config.parallel = enabled;
        self
    }

    /// Set number of threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = Some(threads);
        self
    }

    /// Enable caching
    pub fn with_cache(mut self, cache_dir: impl AsRef<Path>) -> Self {
        self.config.enable_cache = true;
        self.config.cache_dir = Some(cache_dir.as_ref().to_path_buf());
        self
    }

    /// Enable checkpointing
    pub fn with_checkpoints(mut self, interval: Duration) -> Self {
        self.config.checkpoint = true;
        self.config.checkpoint_interval = interval;
        self
    }

    /// Set memory limit
    pub fn memory_limit(mut self, bytes: usize) -> Self {
        self.config.max_memory = Some(bytes);
        self
    }

    /// Add a transformation stage
    pub fn transform<T, U, F>(mut self, name: &str, f: F) -> PipelineBuilder<I, U>
    where
        T: 'static + Send + Sync,
        U: 'static + Send + Sync,
        F: Fn(T) -> Result<U> + Send + Sync + 'static,
    {
        self.stages.push(function_stage(name, f));
        PipelineBuilder {
            stages: self.stages,
            config: self.config,
            _input: self._input,
            _output: PhantomData,
        }
    }

    /// Add a filter stage
    pub fn filter<T, F>(mut self, name: &str, predicate: F) -> Self
    where
        T: 'static + Send + Sync + Clone,
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let stage = function_stage(name, move |input: T| {
            if predicate(&input) {
                Ok(input)
            } else {
                Err(IoError::Other("Filtered out".to_string()))
            }
        });
        self.stages.push(stage);
        self
    }

    /// Add a map stage for collections
    pub fn map_each<T, U, F>(mut self, name: &str, f: F) -> PipelineBuilder<I, Vec<U>>
    where
        T: 'static + Send + Sync,
        U: 'static + Send + Sync,
        F: Fn(T) -> Result<U> + Send + Sync + 'static + Clone,
        O: IntoIterator<Item = T>,
    {
        let stage = function_stage(name, move |input: O| {
            let results: Result<Vec<U>> = input.into_iter().map(|item| f.clone()(item)).collect();
            results
        });
        self.stages.push(stage);
        PipelineBuilder {
            stages: self.stages,
            config: self.config,
            _input: self._input,
            _output: PhantomData,
        }
    }

    /// Add a custom stage
    pub fn stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Add a side effect stage (doesn't transform data)
    pub fn tap<T, F>(mut self, name: &str, f: F) -> Self
    where
        T: 'static + Send + Sync + Clone,
        F: Fn(&T) -> Result<()> + Send + Sync + 'static,
    {
        let stage = function_stage(name, move |input: T| {
            f(&input)?;
            Ok(input)
        });
        self.stages.push(stage);
        self
    }

    /// Add a stage that logs data
    pub fn inspect<T>(mut self, name: &str) -> Self
    where
        T: 'static + Send + Sync + Clone + std::fmt::Debug,
    {
        let name_owned = name.to_string();
        let stage = function_stage(name, move |input: T| {
            println!("[{name_owned}] {input:?}");
            Ok(input)
        });
        self.stages.push(stage);
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Pipeline<I, O> {
        Pipeline {
            stages: self.stages,
            config: self.config,
            _input: PhantomData,
            _output: PhantomData,
        }
    }
}

/// Builder for branching pipelines
pub struct BranchingPipelineBuilder<I> {
    branches: Vec<(String, Box<dyn PipelineStage>)>,
    selector: Box<dyn Fn(&I) -> String + Send + Sync>,
    config: PipelineConfig,
}

impl<I> BranchingPipelineBuilder<I>
where
    I: 'static + Send + Sync,
{
    /// Create a new branching pipeline builder
    pub fn new<F>(selector: F) -> Self
    where
        F: Fn(&I) -> String + Send + Sync + 'static,
    {
        Self {
            branches: Vec::new(),
            selector: Box::new(selector),
            config: PipelineConfig::default(),
        }
    }

    /// Add a branch
    pub fn branch<O, P>(mut self, name: &str, pipeline: Pipeline<I, O>) -> Self
    where
        O: 'static + Send + Sync,
    {
        self.branches.push((
            name.to_string(),
            Box::new(BranchStage {
                name: name.to_string(),
                pipeline: Box::new(pipeline),
            }),
        ));
        self
    }

    /// Build the branching pipeline
    pub fn build<O>(self) -> Pipeline<I, O>
    where
        O: 'static + Send + Sync,
    {
        Pipeline::new().add_stage(Box::new(BranchingStage {
            branches: self.branches.into_iter().collect(),
            selector: self.selector,
        }))
    }
}

struct BranchStage {
    name: String,
    pipeline: Box<dyn Any + Send + Sync>,
}

impl PipelineStage for BranchStage {
    fn execute(
        &self,
        input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        // For now, we execute a simple pass-through with branch metadata
        let mut output = input;
        output.metadata.set("branch_executed", self.name.clone());
        output
            .metadata
            .set("branch_timestamp", chrono::Utc::now().to_rfc3339());
        Ok(output)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "branch".to_string()
    }
}

struct BranchingStage<I> {
    branches: HashMap<String, Box<dyn PipelineStage>>,
    selector: Box<dyn Fn(&I) -> String + Send + Sync>,
}

impl<I> PipelineStage for BranchingStage<I>
where
    I: 'static + Send + Sync,
{
    fn execute(
        &self,
        input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        let typed_input = input
            .data
            .downcast_ref::<I>()
            .ok_or_else(|| IoError::Other("Type mismatch in branching stage".to_string()))?;

        let branch_name = (self.selector)(typed_input);

        if let Some(branch) = self.branches.get(&branch_name) {
            branch.execute(input)
        } else {
            Err(IoError::Other(format!("Unknown branch: {}", branch_name)))
        }
    }

    fn name(&self) -> String {
        "branching".to_string()
    }
}

/// Builder for parallel pipelines
pub struct ParallelPipelineBuilder<I, O> {
    pipelines: Vec<Pipeline<I, O>>,
    combiner: Box<dyn Fn(Vec<O>) -> Result<O> + Send + Sync>,
    config: PipelineConfig,
}

impl<I, O> ParallelPipelineBuilder<I, O>
where
    I: 'static + Send + Sync + Clone,
    O: 'static + Send + Sync,
{
    /// Create a new parallel pipeline builder
    pub fn new<F>(combiner: F) -> Self
    where
        F: Fn(Vec<O>) -> Result<O> + Send + Sync + 'static,
    {
        Self {
            pipelines: Vec::new(),
            combiner: Box::new(combiner),
            config: PipelineConfig::default(),
        }
    }

    /// Add a parallel pipeline
    pub fn pipeline(mut self, pipeline: Pipeline<I, O>) -> Self {
        self.pipelines.push(pipeline);
        self
    }

    /// Build the parallel pipeline
    pub fn build(self) -> Pipeline<I, O> {
        Pipeline::new().add_stage(Box::new(ParallelStage {
            pipelines: self.pipelines,
            combiner: self.combiner,
        }))
    }
}

struct ParallelStage<I, O> {
    pipelines: Vec<Pipeline<I, O>>,
    combiner: Box<dyn Fn(Vec<O>) -> Result<O> + Send + Sync>,
}

impl<I, O> PipelineStage for ParallelStage<I, O>
where
    I: 'static + Send + Sync + Clone,
    O: 'static + Send + Sync,
{
    fn execute(
        &self,
        input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        let typed_input = input
            .data
            .downcast::<I>()
            .map_err(|_| IoError::Other("Type mismatch in parallel stage".to_string()))?;

        // Execute pipelines in parallel
        let results: Result<Vec<O>> = self
            .pipelines
            .par_iter()
            .map(|pipeline| pipeline.execute((*typed_input).clone()))
            .collect();

        let combined = (self.combiner)(results?)?;

        Ok(PipelineData {
            data: Box::new(combined) as Box<dyn Any + Send + Sync>,
            metadata: input.metadata,
            context: input.context,
        })
    }

    fn name(&self) -> String {
        "parallel".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder() {
        let pipeline: Pipeline<i32, String> = PipelineBuilder::<i32, String>::new()
            .transform("double", |x: i32| Ok(x * 2))
            .transform("to_string", |x: i32| Ok(x.to_string()))
            .build();

        let result = pipeline.execute(21).expect("Operation failed");
        assert_eq!(result, "42");
    }

    #[test]
    fn test_pipeline_with_filter() {
        let pipeline: Pipeline<Vec<i32>, Vec<i32>> = PipelineBuilder::<Vec<i32>, Vec<i32>>::new()
            .transform("filter_even", |nums: Vec<i32>| {
                Ok(nums.into_iter().filter(|&x| x % 2 == 0).collect())
            })
            .build();

        let result = pipeline
            .execute(vec![1, 2, 3, 4, 5, 6])
            .expect("Operation failed");
        assert_eq!(result, vec![2, 4, 6]);
    }
}
