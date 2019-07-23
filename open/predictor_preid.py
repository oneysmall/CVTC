import argparse
import csv
import glob
import os
import shutil
import sys
import scipy.misc
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.python.estimator.estimator import _load_global_step_from_checkpoint_dir

from datasets.DatasetFactory import DatasetFactory
from helper.model_helper import get_model_function, get_input_function
from nets import nets_factory

slim = tf.contrib.slim


def start_prediction(data_directory, dataset_name, model_dir, network_name, batch_size, batch_threads, num_classes, distractors):
    dataset_factory = DatasetFactory(dataset_name=dataset_name, data_directory=data_directory, augment=False, num_classes=num_classes)
    
    run_config = RunConfig(keep_checkpoint_max=10, save_checkpoints_steps=None)
    # Instantiate Estimator
    estimator = tf.estimator.Estimator(model_fn=get_model_function(model_dir, network_name, dataset_factory.get_dataset('train').num_classes(), dataset_name=dataset_name),model_dir=model_dir,config=run_config,params={})
    
    image_size = nets_factory.get_input_size(network_name)
    run_prediction_and_evaluation(batch_size, batch_threads, dataset_factory, estimator, image_size, distractors)


def run_prediction_and_evaluation(batch_size, batch_threads, dataset_factory, estimator, image_size, distractors=False):
    output_directory = get_prediction_directory(estimator)  #copy_checkpoint=output_directory_Veid/predictions
    
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    copy_checkpoint(estimator.model_dir, output_directory)  #estimator.model_dir=output_directory_Veid/
    
    print('Starting feature vector generation...')
    
    if distractors:
        run_prediction_and_store_features(dataset_factory, batch_size, batch_threads, estimator, output_directory, 'distractors', image_size)
    
    # run_prediction_and_store_features(dataset_factory, batch_size, batch_threads, estimator, output_directory, 'train', image_size)
    run_prediction_and_store_features(dataset_factory, batch_size, batch_threads, estimator, output_directory, 'test', image_size)
    run_prediction_and_store_features(dataset_factory, batch_size, batch_threads, estimator, output_directory, 'query', image_size)
    
    print('Finished feature vector generation.')
    
    print('Running Matlab evaluation...')
    evaluation_result = run_matlab_evaluation(output_directory)
    save_matlab_evaluation(output_directory, evaluation_result)
    
    print('Finished Matlab evaluation.')
    print(evaluation_result)
    return evaluation_result
    


def get_prediction_directory(estimator):
    return os.path.join(estimator.model_dir, "predictions")


def get_best_prediction_directory(estimator):
    return os.path.join(estimator.model_dir, "predictions-best")


def copy_checkpoint(model_dir, output_directory):
    print('Copying current checkpoint')
    shutil.copyfile(os.path.join(model_dir, 'checkpoint'), os.path.join(output_directory, 'checkpoint'))
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    
    for file in glob.glob(latest_checkpoint + '*'):
        shutil.copy(file, output_directory)



def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def run_prediction_and_store_features(dataset_factory, batch_size, batch_threads, estimator, output_directory, dataset_part, image_size):
    dataset = dataset_factory.get_dataset(dataset_part)
    
    output_directory = os.path.join(output_directory, dataset_part)
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    
    print('\n\nRunning Prediction for %s' % dataset_part)
    input_function = get_input_function(dataset, batch_size, batch_threads, False, image_size)
    
    predicted = estimator.predict(input_fn=input_function)
    num_samples = dataset.get_number_of_samples()
    with open(output_directory + '/features.csv', 'w', newline='') as features_file, open(output_directory + '/labels.csv', 'w', newline='') as labels_file, \
    open(output_directory + '/cameras.csv', 'w', newline='') as cameras_file, open(output_directory+'/names.csv', 'w', newline='') as file_names_file, \
    open(output_directory + '/insertLogits.csv', 'w',  newline='') as insert_features_file:
    
        features_writer = csv.writer(features_file, delimiter=',')
        labels_writer = csv.writer(labels_file, delimiter=',')
        cameras_writer = csv.writer(cameras_file, delimiter=',')
        file_names_writer = csv.writer(file_names_file, delimiter=',')

        insert_features_writer = csv.writer(insert_features_file, delimiter=',')
        
        for sample, prediction in enumerate(predicted):
            if (sample + 1) % batch_size == 0:
                sys.stdout.write('\r>> Processed %d samples of %d' % (sample + 1, num_samples))
                sys.stdout.flush()
            pre_logits = prediction['pre_logits']
            features_writer.writerow(np.squeeze(pre_logits))

            actual_labels = prediction['actual_labels']
            labels_writer.writerow([actual_labels])
            
            file_names = prediction['file_names'].decode('UTF-8')
            file_names_writer.writerow([file_names])

    print('\nFinished Prediction %s' % dataset_part)


best_result = None


def check_init_best_result(estimator):
    global best_result
    if best_result is None:
        best_result = load_matlab_evaluation(get_best_prediction_directory(estimator))
        if best_result is None:
            best_result = MatlabEvaluationResult(mAP=0, rank1=0, rank5=0, rank10=0, rank50=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Specify the folder with the images to be trained and evaluated', dest='data_directory')
    parser.add_argument('--dataset-name', help='The name of the dataset')
    parser.add_argument('--batch-size', help='The batch size', type=int, default=16)
    parser.add_argument('--batch-threads', help='The number of threads to be used for batching', type=int, default=4)
    parser.add_argument('--model-dir', help='The model to be loaded')
    parser.add_argument('--network-name', help='Name of the network')
    parser.add_argument('--num-classes', help='Number of classes', type=int, default=None)
    parser.add_argument('--distractors', help='Should distractors be predicted (only works for market1501)', action='store_true')
    args = parser.parse_args()
    
    print('Running with command line arguments:')
    print(args)
    print('\n\n')

    # tf.logging.set_verbosity(tf.logging.INFO)
    
    start_prediction(args.data_directory, args.dataset_name, args.model_dir, args.network_name, args.batch_size, args.batch_threads, args.num_classes, args.distractors)
    
    print('Exiting ...')


if __name__ == '__main__':
    	main()
