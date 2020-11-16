use greenglas::{Image, Word};
use greenglas::transformer::Transformer;

use coaster::frameworks::native::get_native_backend;
use coaster::tensor::SharedTensor;

use std::rc::Rc;
use std::sync::{Arc, RwLock};

use juice::layer::{LayerConfig, LayerType};
use juice::solver::{SolverConfig, Solver, ConfusionMatrix};
use juice::layers::loss::negative_log_likelihood::NegativeLogLikelihoodConfig;
use juice::layers::{
    utility::reshape::ReshapeConfig,
    container::sequential::SequentialConfig,
};
use juice::layers::common::{
    convolution::ConvolutionConfig,
    pooling::{PoolingConfig, PoolingMode},
    linear::LinearConfig,
};

fn main() {
    let batch_size = 30;
    let pixel_dim = 64;
    let learning_rate = 0.001;
    let momentum = 0f32;
    let training_size = 12000;

    let mut model = SequentialConfig::default();
    model.add_layer(LayerConfig::new("reshape", ReshapeConfig::of_shape(&[1, pixel_dim, pixel_dim]),));
    model.add_layer(LayerConfig::new("conv", ConvolutionConfig { num_output: 32, filter_shape: vec![3], padding: vec![0], stride: vec![1] }));
    model.add_layer(LayerConfig::new("pooling", PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] }));
    model.add_layer(LayerConfig::new("linear1", LinearConfig { output_size: 500 }));
    model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
    model.add_layer(LayerConfig::new("linear1", LinearConfig { output_size: 15 }));
    
    let mut classifier = SequentialConfig::default();
    classifier.add_input("network_out", &[batch_size, 15]);
    classifier.add_input("label", &[batch_size, 1]);

    let nll_layer = NegativeLogLikelihoodConfig { num_classes: 15 };
    let nll = LayerConfig::new("nll", LayerType::NegativeLogLikelihood(nll_layer));
    classifier.add_layer(nll);

    let backend = Rc::new(get_native_backend());
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum,
        ..SolverConfig::default()
    };

    solver_cfg.network = LayerConfig::new("network", model);
    solver_cfg.objective = LayerConfig::new("classifier", classifier);
    let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    let mut classification_elevator = ConfusionMatrix::new(15);
    classification_elevator.set_capacity(Some(1000));

    let input = SharedTensor::<f32>::new(&[batch_size, pixel_dim, pixel_dim]);
    let inp_lock = Arc::new(RwLock::new(input));

    let label = SharedTensor::<f32>::new(&[batch_size, 1]);
    let label_lock = Arc::new(RwLock::new(label));

    for _ in 0..(training_size / batch_size as u32) {
        for _ in 0..batch_size
        {
            let image = Image::from_path("dataset/data/input_1_1_1.jpg");
            let label = Word::new(String::from("label"));

            let mut input_tensor = inp_lock.write().unwrap();
            let mut label_tensor = label_lock.write().unwrap();

            *input_tensor = image.transform(&[pixel_dim, pixel_dim, 1]).unwrap();
            *label_tensor = label.transform(&[1]).unwrap();
        }

        let infered_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());

        let mut infered = infered_out.write().unwrap();
        let predictions = classification_elevator.get_predictions(&mut infered);

        println!("prediction : {:?}", predictions);
    }
}
