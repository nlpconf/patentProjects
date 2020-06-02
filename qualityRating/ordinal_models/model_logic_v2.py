import tensorflow as tf
import numpy as np
from .baseModel import BaseModel
from .model_v2 import Model
from sklearn.metrics import precision_recall_fscore_support
from qualityRating.utilities.utils import evalReport,data_iterator
from qualityRating.utilities.utils import calmacroMAE
import datetime
import shutil
import os
import time

def make_prediction(probabilities):
    probabilities = np.array(probabilities)
    range_prob = np.ones((probabilities.shape[0], probabilities.shape[-1] + 1))
    range_prob[:, 1:] = probabilities
    # print(range_prob)
    range_prob[:, :-1] = range_prob[:, :-1] - probabilities
    pred = np.argmax(range_prob, axis=1)
    #print(probabilities[:5, :])
    #print(range_prob[:5, :])
    return range_prob, pred
class Model_v2(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)

    def buildModel(self):
        self.model = Model(self.config,self.vocab.wordVectors)
        self.model.build()
        self.loss = self.model.loss
        self.add_optimizer()
        self.initialize_session()
        self.add_summary()

    def train_step(self,batch,e):
        """
        A single training step
        """
        #print(batch['input_x'].shape)

        #batch_doc = np.array(batch['content'])
        #batch_input_actual_num_sents = np.array(batch['content_actual_num_sents'])
        #batch_similar_docs_num_sents = np.array(batch['content_similar_docs_actual_num_sents'])

        batch_cls = np.array(batch['target'])
        #print(batch_cls)
        '''
        print(type(self.model.input_cls))
        print(type(batch_cls),batch_cls)
        print('batch_cls',batch_cls.shape)
        print('batch_doc',batch_doc.shape)
        print('doc_actual_num_sents',batch_input_actual_num_sents.shape)

        print('batch_similar_docs', batch_similar_docs.shape)
        print('batch_similar_docs_num_sents', batch_similar_docs_num_sents.shape)
        '''
        batch_inputs = []
        batch_input_actual_num_sents = []
        batch_similar_actual_num_sents = []
        batch_similar_docs = []
        for name in self.config.text_field_names:
            batch_inputs.append(batch[name])
            batch_similar_docs.append(batch[name + '_similar_docs' ])
            batch_input_actual_num_sents.append(batch[name + '_actual_num_sents'])
            batch_similar_actual_num_sents.append(batch[name + '_similar_docs_actual_num_sents'])

        feed_dict = {
            self.model.cnn_dropout_keep_prob: self.config.cnn_dropout_keep_prob,
            self.model.input_cls:batch_cls
        }
        feed_dict.update({ph: data for ph, data in zip(self.model.input_doc, batch_inputs)})
        feed_dict.update({ph: data for ph, data in zip(self.model.similar_docs, batch_similar_docs)})
        feed_dict.update({ph: data for ph, data in zip(self.model.doc_actual_num_sents, batch_input_actual_num_sents)})
        feed_dict.update({ph: data for ph, data in zip(self.model.similar_doc_actual_num_sents, batch_similar_actual_num_sents)})

        start_time = time.time()
        _, step, summaries, loss= self.sess.run(
            [self.train_op, self.global_step, self.train_summary_op, self.loss],
            feed_dict)

        end_time = time.time()
        time_str = datetime.datetime.now().isoformat()
        print("\rtraining:{}:epoch {} step {}, loss {:g}, time_consumed {}".format(time_str, e, step, loss,
                                                                                   end_time - start_time))
        self.train_summary_writer.add_summary(summaries, step)
        '''
        doc_features_shape = self.sess.run(self.model.doc_features_shape,feed_dict)
        print('doc_features_shape',doc_features_shape)
        return 0
        '''
        return loss


    def eval_step(self,data,ids,all_data):
        """
        Evaluates model on a eval set
        """
        doc_cls_predictions = []
        doc_cls_probabilities =[]
        labels = []
        for batch in data_iterator(data,all_data,self.config.text_field_names,ids = ids,batch_size=self.config.batch_size, shuffle=False):

            batch_cls = np.array(batch['target'])
            labels.extend(batch['label'])

            batch_inputs = []
            batch_input_actual_num_sents = []
            batch_similar_actual_num_sents = []
            batch_similar_docs = []
            for name in self.config.text_field_names:
                batch_inputs.append(batch[name])
                batch_similar_docs.append(batch[name + '_similar_docs'])
                batch_input_actual_num_sents.append(batch[name + '_actual_num_sents'])
                batch_similar_actual_num_sents.append(batch[name + '_similar_docs_actual_num_sents'])

            feed_dict = {
                self.model.cnn_dropout_keep_prob: 1.0,
                self.model.input_cls: batch_cls
            }
            feed_dict.update({ph: data for ph, data in zip(self.model.input_doc, batch_inputs)})
            feed_dict.update({ph: data for ph, data in zip(self.model.similar_docs, batch_similar_docs)})
            feed_dict.update(
                {ph: data for ph, data in zip(self.model.doc_actual_num_sents, batch_input_actual_num_sents)})
            feed_dict.update(
                {ph: data for ph, data in zip(self.model.similar_doc_actual_num_sents, batch_similar_actual_num_sents)})
            step, summaries, doc_cls_prob= self.sess.run(
                [self.global_step, self.dev_summary_op, self.model.doc_cls_probabilities],
                feed_dict)

            _, doc_cls_pred = make_prediction(doc_cls_prob)
            doc_cls_probabilities.extend(doc_cls_prob)
            doc_cls_predictions.extend(doc_cls_pred.tolist())

        doc_precision, doc_recall, doc_f1_score, status = precision_recall_fscore_support(labels, np.array(doc_cls_predictions),
                                                                                labels=range(0, self.config.num_classes),
                                                                                pos_label=None,
                                                                                average='macro')

        mae,_ = calmacroMAE(labels, np.array(doc_cls_predictions), self.config.num_classes)
        print(doc_precision, doc_recall, doc_f1_score,mae)

        return doc_precision, doc_recall, doc_f1_score,mae

    def saveModel(self, export_path):
        print('Exporting trained model to', export_path)
        if self.builder == None:
            if os.path.exists(export_path):
                shutil.rmtree(export_path)
            savedModel_path = export_path
            self.config.model_path = savedModel_path
            self.builder = tf.saved_model.builder.SavedModelBuilder(savedModel_path)

            input_tensors = {}
            for i,name in enumerate(self.config.text_field_names):
                input_tensors[name + '_input_doc'] = tf.saved_model.utils.build_tensor_info(self.model.input_doc[i])
                input_tensors[name + '_similar_docs'] = tf.saved_model.utils.build_tensor_info(self.model.similar_docs[i])
                input_tensors[name + '_actual_num_sents'] = tf.saved_model.utils.build_tensor_info(self.model.doc_actual_num_sents[i])
                input_tensors[name + '_similar_docs_actual_num_sents'] = tf.saved_model.utils.build_tensor_info(self.model.similar_doc_actual_num_sents[i])

            #doc_cls_predictions_tensor_info = tf.saved_model.utils.build_tensor_info(self.model.doc_cls_probabilities)
            doc_cls_probabilities_tensor_info = tf.saved_model.utils.build_tensor_info(self.model.doc_cls_probabilities)

            output_tensors = {
                        'doc_cls_probabilities':doc_cls_probabilities_tensor_info}

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=input_tensors,
                    outputs=output_tensors,
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            #legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            self.builder.add_meta_graph_and_variables(
                self.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_patent':
                        prediction_signature,
                })
                #legacy_init_op=legacy_init_op)

        self.builder.save()
        self.builder = None

        print('Done exporting!')

    def eval(self,test_data,all_data,test_ids, config):
        doc_cls_predictions,_ =self.predict(test_data,all_data,test_ids,config,config.model_path)
        labels=[]
        for batch in data_iterator(test_data, all_data, config.text_field_names,ids = test_ids, batch_size=config.batch_size,
                                   shuffle=False):
            labels+=batch['label']
        doc_cls_reports = evalReport(labels, doc_cls_predictions,config, config.num_classes)

        return doc_cls_reports
    @staticmethod
    def predict(test_data,all_data,test_ids,config,model_path):
        graph = tf.Graph()
        with graph.as_default():
            start_time = time.time()
            session_conf = tf.ConfigProto(
                allow_soft_placement=config.allow_soft_placement,
                log_device_placement=config.log_device_placement)
            sess = tf.Session(config=session_conf)
            end_time = time.time()
            print("load session time : %f" % (end_time - start_time))

            print('')
            with sess.as_default():
                start_time = time.time()

                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
                end_time = time.time()

                print("load model time : %f" % (end_time - start_time))

                input_tensors = {}

                for name in config.text_field_names:
                    input_tensors[name + '_input_doc'] = graph.get_tensor_by_name(name +'_input_doc'+":0")
                    input_tensors[name + '_similar_docs'] = graph.get_tensor_by_name(name + '_similar_docs' +":0" )
                    input_tensors[name + '_actual_num_sents'] = graph.get_tensor_by_name(name + '_doc_actual_num_sents' +":0" )
                    input_tensors[name + '_similar_docs_actual_num_sents'] =graph.get_tensor_by_name(name + '_similar_doc_actual_num_sents' + ":0")
                #input_tensors['cnn_dropout_keep_prob'] = graph.get_tensor_by_name("cnn_dropout_keep_prob:0")

                #doc_cls_predictions = graph.get_operation_by_name("predictions").outputs[0]
                doc_cls_probabilities = graph.get_operation_by_name("probabilities").outputs[0]


                cls_predictions= []
                cls_probabilities =[]
                for batch in data_iterator(test_data, all_data,config.text_field_names,ids = test_ids, batch_size=config.batch_size,
                                           shuffle=False):

                    batch_inputs = []
                    batch_input_actual_num_sents = []
                    batch_similar_actual_num_sents = []
                    batch_similar_docs = []
                    input_data = {}
                    for name in config.text_field_names:
                        input_data[name + '_input_doc'] = batch[name]
                        input_data[name + '_similar_docs'] = batch[name + '_similar_docs']
                        input_data[name + '_actual_num_sents'] = batch[name + '_actual_num_sents']
                        input_data[name + '_similar_docs_actual_num_sents'] = batch[name + '_similar_docs_actual_num_sents']

                    feed_dict = {}
                    for key in input_tensors:
                        feed_dict[input_tensors[key]] = input_data[key]

                    doc_cls_probs= sess.run(doc_cls_probabilities,
                                                         feed_dict)
                    _, doc_cls_pred = make_prediction(doc_cls_probs)
                    cls_predictions.extend(doc_cls_pred.tolist())
                    cls_probabilities.extend(doc_cls_probs)
        tf.reset_default_graph()

        return cls_predictions,cls_probabilities


    @staticmethod
    def output_similarity_vectors(test_data, all_data, test_ids, config, model_path):
        graph = tf.Graph()
        with graph.as_default():
            start_time = time.time()
            session_conf = tf.ConfigProto(
                allow_soft_placement=config.allow_soft_placement,
                log_device_placement=config.log_device_placement)
            sess = tf.Session(config=session_conf)
            end_time = time.time()
            print("load session time : %f" % (end_time - start_time))

            print('')
            with sess.as_default():
                start_time = time.time()

                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
                end_time = time.time()

                print("load model time : %f" % (end_time - start_time))

                input_tensors = {}

                for name in config.text_field_names:
                    input_tensors[name + '_input_doc'] = graph.get_tensor_by_name(name + '_input_doc' + ":0")
                    input_tensors[name + '_similar_docs'] = graph.get_tensor_by_name(name + '_similar_docs' + ":0")
                    input_tensors[name + '_actual_num_sents'] = graph.get_tensor_by_name(
                        name + '_doc_actual_num_sents' + ":0")
                    input_tensors[name + '_similar_docs_actual_num_sents'] = graph.get_tensor_by_name(
                        name + '_similar_doc_actual_num_sents' + ":0")
                # input_tensors['cnn_dropout_keep_prob'] = graph.get_tensor_by_name("cnn_dropout_keep_prob:0")

                # doc_cls_predictions = graph.get_operation_by_name("predictions").outputs[0]
                #doc_cls_probabilities = graph.get_operation_by_name("probabilities").outputs[0]

                similarity_vectors = []
                for i, name in enumerate(config.text_field_names):
                    if i == 0:
                        similarity_vectors.append(graph.get_tensor_by_name("Mean:0"))
                    else:
                        similarity_vectors.append(graph.get_tensor_by_name("Mean_%d:0" % i))

                predicted_similarity_vectors = []
                for batch in data_iterator(test_data, all_data, config.text_field_names, ids=test_ids,
                                           batch_size=config.batch_size,
                                           shuffle=False):
                    input_data = {}
                    for name in config.text_field_names:
                        input_data[name + '_input_doc'] = batch[name]
                        input_data[name + '_similar_docs'] = batch[name + '_similar_docs']
                        input_data[name + '_actual_num_sents'] = batch[name + '_actual_num_sents']
                        input_data[name + '_similar_docs_actual_num_sents'] = batch[name + '_similar_docs_actual_num_sents']

                    feed_dict = {}
                    for key in input_tensors:
                        feed_dict[input_tensors[key]] = input_data[key]

                    batch_similarity_vectors = sess.run(similarity_vectors,
                                             feed_dict)
                    batch_similarity_vectors = np.stack(batch_similarity_vectors, axis=1)
                    print(batch_similarity_vectors.shape)
                    predicted_similarity_vectors.extend(batch_similarity_vectors.tolist())
        tf.reset_default_graph()

        return predicted_similarity_vectors
