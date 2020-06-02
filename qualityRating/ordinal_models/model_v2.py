import tensorflow as tf
#from .attention import attention
class Model(object):
    def __init__(self, config,vectors):
        self.max_num_sents = config.max_num_sents
        self.max_sent_length = config.max_sent_length
        self.num_classes = config.num_classes -1
        self.num_simlar_docs = config.num_simlar_docs
        self.num_concepts = config.num_concepts
        self.concept_dimesions = config.concept_dimesions
        self.vectors = vectors
        self.word_embedding_size = config.word_embedding_size
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.text_field_names = config.text_field_names
        self.similarity_text_field_names = config.similarity_text_field_names
        self.use_loss = config.use_loss
        self.threshold = config.threshold
        self.l2_reg_lambda = config.l2_reg_lambda
        #self.use_noise = config.use_noise
        self.input_doc = []
        self.similar_docs = []
        self.doc_actual_num_sents = []
        self.similar_doc_actual_num_sents = []
        for name in self.text_field_names:
            self.input_doc.append(tf.placeholder(tf.int32, [None, self.max_num_sents[name], self.max_sent_length[name]], name=name + '_input_doc'))
            self.similar_docs.append(tf.placeholder(tf.int32, [None,self.num_simlar_docs, self.max_num_sents[name], self.max_sent_length[name]], name=name +'_similar_docs' ))
            self.doc_actual_num_sents.append(tf.placeholder(tf.int32, [None], name=name+'_doc_actual_num_sents'))
            self.similar_doc_actual_num_sents.append(tf.placeholder(tf.int32, [None,self.num_simlar_docs], name=name+'_similar_doc_actual_num_sents'))

        self.input_cls = tf.placeholder(tf.float32, [None,self.num_classes], name='input_cls')
        self.cnn_dropout_keep_prob = tf.placeholder_with_default(1.0,[], name='cnn_dropout_keep_prob')
        #self.attention_keep_prob = tf.placeholder(tf.float32, name='attention_keep_prob')
        #self.rnn_output_keep_prob = tf.placeholder(tf.float32, name='rnn_output_keep_prob')
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)
        self.fine_tune_word_embedding=config.fine_tune_word_embedding

        self.wv_initial = tf.constant(self.vectors, dtype=tf.float32)

    def add_embedding_layer(self,input):


        # Embedding layer
        with tf.variable_scope('embedding',reuse=tf.AUTO_REUSE):
            wordVectors = tf.get_variable('word_vectors', initializer=self.wv_initial,trainable=self.fine_tune_word_embedding)
            embedded_words = tf.nn.embedding_lookup(wordVectors, input)
        return embedded_words


    def add_cnn_layer(self, input, length, filter_sizes, input_dimension, scope):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('%s-conv-maxpool-%s' % (scope,filter_size), reuse=tf.AUTO_REUSE):
                # Convolution Layer
                #filter_shape = [filter_size, self.word_embedding_size + self.position_embedding_size, 1, self.num_filters]
                filter_shape = [filter_size, input_dimension , 1,
                                self.num_filters]
                W = tf.get_variable(name='W',initializer = tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable(name='b',initializer = tf.constant(0.1, shape=[self.num_filters]))
                conv = tf.nn.conv2d(
                    input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)
                #print('pooled shape', pooled.get_shape())
        # Combine all the pooled features
        num_features = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_features])

        # Add dropout
        with tf.variable_scope('cnn-dropout-%s'%scope):
            cnn_drop = tf.nn.dropout(h_pool_flat, self.cnn_dropout_keep_prob)
        return cnn_drop,num_features


    def sentence_features(self,input,max_sent_length):

        #sentence_input = tf.reshape(input,[-1,self.max_sent_length,self.word_embedding_size])
        sentence_input = tf.expand_dims(input,-1)#tf.slice(self.cnn_concat_word_pos_embedded_expanded, [0,i,0,0,0],[0,1,self.sequence_length,self.word_embedding_size,1])
        sentence_features, sentence_num_features = self.add_cnn_layer(sentence_input,max_sent_length,self.filter_sizes,self.word_embedding_size,'sentence' )
        #sentence_feature = tf.reshape(sentence_feature,[-1,self.max_num_sents,sentence_num_features] )
        return sentence_features,sentence_num_features
    def mask_sequence(self,input,max_length, input_actual_num,last_dimesion_size):

        mask = tf.to_float(tf.sequence_mask(input_actual_num, max_length))
        #print('input',input.get_shape())
        #print('mask',mask.get_shape())
        mask = tf.tile(tf.expand_dims(mask,-1),[1,1,last_dimesion_size])
        self.mask_shape = tf.shape(mask)
        masked = input * mask
        return masked
    def doc_features(self,input,max_num_sents):

        doc_input = tf.expand_dims(input,-1)

        doc_feature,_ = self.add_cnn_layer(doc_input,max_num_sents,self.filter_sizes,self.num_filters * len(self.filter_sizes),'doc')
        return doc_feature
        #print('doc_features',self.doc_features.get_shape())

    #def mask(self, input):

    def add_fc_layer(self, input, input_size, output_size, scope):

        #self.feature = self.cnn_drop
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                'W',
                shape=[input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b', initializer=tf.constant(0.1, shape=[output_size]))
            vectors = tf.nn.xw_plus_b(input, W, b, name='scores')
        return vectors, W, b
    def convert_to_concept_space(self,num_to_convert, input, input_size, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            """
            W = tf.get_variable(
                'W',
                shape=[num_to_convert, input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer())

            concept_vector = tf.einsum('bsij,cjk->bscik', input, W)
            """
            W = tf.get_variable(
                'W',
                shape=[num_to_convert, input_size],
                initializer=tf.contrib.layers.xavier_initializer())
            concept_vector = tf.einsum('bsi,ci->bsci', input, W)

        return concept_vector
    def cal_similarities(self,input_1,input_2):
        #dot product
        #similarity = tf.matmul(input_1, input_2, transpose_b=True)
        print('similar_doc_concepts[i]', input_2.get_shape())
        print('doc_concepts', input_1.get_shape())
        #similarity = tf.einsum('bsc,bsc->bs', input_1, input_2)
        similarity = tf.einsum('bsnc,bsnc->bsn', input_1, input_2)
        return similarity

    def make_prediction(self,probabilities,threshold):
        pred = tf.where(tf.greater(probabilities,threshold),tf.ones_like(probabilities,dtype=tf.float32),tf.zeros_like(self.input_cls,dtype=tf.float32))
        predictions = tf.identity(tf.reduce_sum(pred,axis = 1) - 1,name='predictions')
        return predictions
    def _doc_prediction(self,input,input_size,num_classes,scope):
        scores, W, b = self.add_fc_layer(input, input_size, num_classes, scope)
        probabilities = tf.nn.sigmoid(scores,name='probabilities')
        #predictions = self.make_prediction(probabilities,self.threshold)
        return scores, probabilities,W,b


    def doc_prediction(self,input,input_size,scope):

        self.doc_cls_scores,self.doc_cls_probabilities,W,b = self._doc_prediction(input,
                                                                            input_size,
                                                                        self.num_classes,scope)
        self.l2_loss += tf.nn.l2_loss(W)
        self.l2_loss += tf.nn.l2_loss(b)
        #print(self.doc_cls_scores)
    def add_loss(self):
        if self.use_loss == "cross-entropy":

            with tf.name_scope('loss'):
                doc_losses = tf.constant(0.0)
                #cls = tf.contrib.layers.one_hot_encoding(self.input_cls,num_classes=self.num_classes)
                doc_losses += tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_cls,logits=self.doc_cls_scores)
                self.doc_losses = tf.reduce_mean(doc_losses)

                self.loss = self.doc_losses + self.l2_reg_lambda * self.l2_loss

        elif self.use_loss == "square_error":
            # Calculate mean absolute error
            with tf.name_scope('loss'):
                #diff = tf.subtract(self.predictions, tf.argmax(self.input_y, 1))
                losses = tf.losses.mean_squared_error(self.input_cls, self.doc_cls_probabilities)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
    def build(self):
        features = []
        for i, name in enumerate(self.text_field_names):
            input_embedded_words = self.add_embedding_layer(self.input_doc[i])
            sentence_input = tf.reshape(input_embedded_words, [-1, self.max_sent_length[name], self.word_embedding_size])
            sentence_features,sentence_num_features = self.sentence_features(sentence_input,self.max_sent_length[name])
            #self.sentence_features_shape = tf.shape(sentence_features)

            sentence_features = tf.reshape(sentence_features,[-1,self.max_num_sents[name],sentence_num_features])

            sentence_features = self.mask_sequence(sentence_features,
                                                   self.max_num_sents[name],
                                                   self.doc_actual_num_sents[i],
                                                   sentence_num_features)


            if self.max_num_sents[name] <= max(self.filter_sizes):
                #features.append(tf.reshape(sentence_features,[-1,sentence_num_features] ))
                doc_features = tf.reshape(sentence_features,[-1,sentence_num_features] )

            else:
                doc_features = self.doc_features(sentence_features, self.max_num_sents[name])

            if name in self.similarity_text_field_names:

                sentence_concepts = self.convert_to_concept_space(self.num_concepts,sentence_features,
                                                                 sentence_num_features,
                                                                 "sentence_concept")
                doc_concepts = tf.reduce_sum(sentence_concepts,axis=1) /tf.cast(self.doc_actual_num_sents[i],dtype=tf.float32)


                similar_doc_input = self.add_embedding_layer(self.similar_docs[i])
                similar_doc_input = tf.reshape(similar_doc_input,[-1,self.max_sent_length[name],self.word_embedding_size])

                similar_doc_sentence_feature, similar_doc_sentence_num_features = self.sentence_features(similar_doc_input,self.max_sent_length[name])
                #similar_doc_sentence_feature, similar_doc_sentence_num_features = tf.reduce_mean(similar_doc_input,axis=1)
                similar_doc_sentence_feature = tf.reshape(similar_doc_sentence_feature,
                                                          [-1,self.max_num_sents[name], similar_doc_sentence_num_features])

                similar_doc_actual_num_sents = tf.reshape(self.similar_doc_actual_num_sents[i], [-1])
                similar_doc_sentence_feature = self.mask_sequence(similar_doc_sentence_feature,
                                                                  self.max_num_sents[name],
                                                                  similar_doc_actual_num_sents,
                                                                  similar_doc_sentence_num_features)

                #similar_doc_sentence_feature = tf.reshape(-1,self.num_simlar_docs,self.max_num_sents,similar_doc_sentence_num_features)

                similar_doc_sentence_concept = self.convert_to_concept_space(self.num_concepts,similar_doc_sentence_feature,
                                                                             similar_doc_sentence_num_features,
                                                                             "sentence_concept")
                similar_doc_sentence_concept = tf.reduce_sum(similar_doc_sentence_concept, axis=1)#sum all sentence

                similar_doc_concepts = tf.reshape(similar_doc_sentence_concept,[-1,self.num_simlar_docs,self.num_concepts,similar_doc_sentence_num_features])
                #print('one_similar_doc_concept',similar_doc_concepts.get_shape())

                doc_concepts = tf.reshape(doc_concepts,[-1,1,self.num_concepts,sentence_num_features])
                doc_concepts = tf.tile(doc_concepts,[1,self.num_simlar_docs,1,1])
                #similar_doc_similarities =self.cal_similarities(doc_concepts,similar_doc_concepts)

                similar_doc_similarities =tf.reduce_sum(tf.multiply(doc_concepts,similar_doc_concepts),axis=3)
                print('similar_doc_similarities',similar_doc_similarities.get_shape())
                average_similar_doc_similarities = tf.reduce_mean(similar_doc_similarities,axis=1)

                print(name, average_similar_doc_similarities)
                '''
                similar_doc_similarities = []
                for i in range(self.num_simlar_docs):
                    #print('one_similar_doc_concept', similar_doc_concepts[:,i,:,:].get_shape())
                    #print('doc_concepts', doc_concepts.get_shape())

                    similar_doc_similarity = self.cal_similarities(doc_concepts,similar_doc_concepts[:,i,:,:])
                    similar_doc_similarities.append(similar_doc_similarity)
                similar_doc_similarities = tf.stack(similar_doc_similarities,axis =1 )
                average_similar_doc_similarities = tf.reduce_mean(similar_doc_similarities,axis=1)
                '''
                final_text_field_features = tf.concat([doc_features,average_similar_doc_similarities],axis = -1)
                features.append(final_text_field_features)
            else:
                features.append(doc_features)

        features = tf.concat(features,axis = -1)
        self.doc_prediction(features, features.get_shape()[-1],'doc_predict')

        #self.doc_prediction(doc_features, doc_features.get_shape()[-1],'doc_predict')


        self.add_loss()

