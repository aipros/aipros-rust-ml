use csv::ReaderBuilder;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{prelude::*, OwnedRepr};
use ndarray_csv::Array2Reader;
use plotlib::{
    grid::Grid,
    page::Page,
    repr::Plot,
    style::{PointMarker, PointStyle},
    view::{ContinuousView, View},
};

fn main() {
    let train = load_data("data/train.csv");
    let test = load_data("data/test.csv");

    let features = train.nfeatures();
    let targets = train.ntargets();

    println!(
        "training with {} samples, testing with {} samples, {} features and {} target",
        train.nsamples(),
        test.nsamples(),
        features,
        targets
    );

    println!("plotting data...");
    plot_data(&train);

    println!("training and testing model...");
    let mut max_accuracy_confusion_matrix = iterate_with_values(&train, &test, 0.01, 100);
    let mut best_threshold = 0.0;
    let mut best_max_iterations = 0;
    let mut threshold = 0.02;

    for max_iterations in (1000..5000).step_by(500) {
        while threshold < 1.0 {
            let confusion_matrix = iterate_with_values(&train, &test, threshold, max_iterations);

            if confusion_matrix.accuracy() > max_accuracy_confusion_matrix.accuracy() {
                max_accuracy_confusion_matrix = confusion_matrix;
                best_threshold = threshold;
                best_max_iterations = max_iterations;
            }
            threshold += 0.01;
        }
        threshold = 0.02;
    }

    println!(
        "most accurate confusion matrix: {:?}",
        max_accuracy_confusion_matrix
    );
    println!(
        "with max_iterations: {}, threshold: {}",
        best_max_iterations, best_threshold
    );
    println!("accuracy {}", max_accuracy_confusion_matrix.accuracy(),);
    println!("precision {}", max_accuracy_confusion_matrix.precision(),);
    println!("recall {}", max_accuracy_confusion_matrix.recall(),);
}

fn iterate_with_values(
    train: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >,
    test: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >,
    threshold: f64,
    max_iterations: u64,
) -> ConfusionMatrix<&'static str> {
    let model = LogisticRegression::default()
        .max_iterations(max_iterations)
        .gradient_tolerance(0.0001)
        .fit(train)
        .expect("can train model");

    let validation = model.set_th