#!/usr/bin/env python
#
# Copyright 2017 The Open Images Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import tensorflow as tf



class TagModel(object):

    def __init__(self, user_top_k, score_threshold):
        """
            Define model files and parameters
            Args:
            - model_files_folder: path to folder with model files
            - user_top_k: number max of tags returned by model
            - score_threshold: proba mini for a tag to be returned
        """
        import os
        print(os.getcwd())
        # model params
        self.user_top_k = user_top_k
        self.score_threshold = score_threshold
        # model files
        self.labelmap_path   = join(os.path.dirname(__file__), "model_files/classes-trainable.txt")
        self.dict_path       = join(os.path.dirname(__file__), "model_files/class-descriptions.csv")
        self.checkpoint_path = join(os.path.dirname(__file__), "model_files/oidv2-resnet_v1_101.ckpt")


    def load(self):
        """
            Call load_label_map() and load_model()
            - load_label_map(): Load objects used for tag mapping
            - load_model(): Load tf sessions used to make predictions
        """
        self.labelmap, self.label_dict = self.load_label_map()
        self.sess, self.input_values, self.predictions = self.load_model()


    def generate_tags(self, image):
        """
            Take an image an generate tag relative to the image
            Args:
            - image: opened image
            Returns:
            - generated_tags: list of tag generated for image.
                              Each tag is an instance of Tag class and has a .name and .score attr
                              Max number of tags is defined by user_top_k
                              Tag's minimum score is defined by score_threshold
        """
        # compute predictions (tags)
        predictions_eval = self.sess.run(
            self.predictions, feed_dict={
                self.input_values: [image]
            })
        # sort predictions by score
        top_k = predictions_eval.argsort()[::-1]
        # take n tags (n = user_top_k)
        if self.user_top_k > 0:
            top_k = top_k[:self.user_top_k]
        # remove tags with score below score_threshold
        if self.score_threshold is not None:
            top_k = [i for i in top_k if predictions_eval[i] >= self.score_threshold]

        # transfom tag id to user friendly tag and store each tag in generated_tags
        generated_tags = []
        for idx in top_k:
            mid = self.labelmap[idx]
            display_name = self.label_dict[mid]
            score = predictions_eval[idx]
            #print('{:04d}: {} - {} (score = {:.2f})'.format(
                #idx, mid, display_name, score))
            tag = f"{display_name}_{score}"
            generated_tags.append(tag)

        return generated_tags

#-----------------------------------------------------------------------------------------
# PRIVATE
#-----------------------------------------------------------------------------------------

    def load_label_map(self):
        """
            Map index to mid label and mid label to user friendly tag
            Returns:
            - labelmap: an index to mid list
            - label_dict: mid to display name dictionary
        """
        labelmap = [line.rstrip() for line in tf.gfile.GFile(self.labelmap_path)]

        label_dict = {}
        for line in tf.gfile.GFile(self.dict_path):
            words = [word.strip(' "\n') for word in line.split(',', 1)]
            label_dict[words[0]] = words[1]

        return labelmap, label_dict


    def load_model(self):
        """
            load model graph and weights
            Returns:
            - sess: tf session used to make predictions
            - inputs_values: input tensor
            - predictions: output tensor
        """
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()
            saver = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
            saver.restore(sess, self.checkpoint_path)

            input_values = g.get_tensor_by_name('input_values:0')
            predictions = g.get_tensor_by_name('multi_predictions:0')

            return sess, input_values, predictions