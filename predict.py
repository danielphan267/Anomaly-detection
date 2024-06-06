from model import Inception_Inflated3d, classifier_i3d_model, build_classifier_model, get_video_clips, preprocess_input_i3d, visualize_predictions
from model import interpolate, extrapolate
import os
import sys
import numpy as np

def run_model(video_path):
    i3d_frame_count = 16
    features_per_bag = 32
    output_folder_i3d = 'F:\Anomaly Detection/static/results'

    model_i3d_feature_extract = Inception_Inflated3d(include_top=False,
                    weights='rgb_imagenet_and_kinetics',
                    input_shape=(16,224,224,3),
                    dropout_prob=0.0,
                    endpoint_logit=True,
                    classes=400)

    output_dir = "F:\Anomaly Detection/Anomaly Recognition/Trained_models/"

    weights_path_i3d = output_dir + 'weights_i3d.mat'
    model_path_i3d = output_dir + 'model_i3d.json'

    model_i3d = classifier_i3d_model()

    model_i3d = build_classifier_model(model_i3d, weights_path_i3d)


    def run_demo_i3d(sample_video_path, video_name):
        # read video
        video_clips, num_frames = get_video_clips(sample_video_path, i3d_frame_count)
        
        print('Number of frames: ', num_frames)
        print("Number of clips in the video : ", len(video_clips))

        # build models
        feature_extractor = model_i3d_feature_extract
        classifier_model = model_i3d

        print("Models initialized")

        # extract features
        rgb_features = []
        for i, clip in enumerate(video_clips):
            clip = np.array(clip)
            if len(clip) < i3d_frame_count:
                continue

            clip = preprocess_input_i3d(clip)
            rgb_feature = feature_extractor.predict(clip)[0]
            rgb_features.append(rgb_feature)

            print("Processed clip : ", i)

        rgb_features = np.array(rgb_features)
        rgb_features = np.reshape(rgb_features, (rgb_features.shape[0], rgb_features.shape[4]))
        rgb_feature_bag = interpolate(rgb_features, features_per_bag)


        # classify using the trained classifier model
        predictions = classifier_model.predict(rgb_feature_bag)
        
        predictions = np.array(predictions).squeeze()

        predictions = extrapolate(predictions, num_frames)

        save_path = os.path.join(output_folder_i3d, video_name + '.gif')

        # visualize predictions
        print('Executed Successfully - ' + video_name + 'saved')
        visualize_predictions(sample_video_path, predictions, save_path)

    video_name = os.path.basename(video_path)
    run_demo_i3d(video_path, video_name)


if __name__ == "__main__":
    video_path = sys.argv[1]
    run_model(video_path)
    
    # sample_video_folder = 'F:\Anomaly Detection/videos'
    # sample_video_list = os.listdir(sample_video_folder)
    # sample_video_list.sort()

    # for sample_video_name in sample_video_list:  
        
    #     sample_video_path = os.path.join(sample_video_folder, sample_video_name)
    #     run_model(sample_video_path)