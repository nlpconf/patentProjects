import tensorflow as tf
from .attention import attention
class Model(object):
    def __init__(self, config,vectors):
        self.max_num_sents = config.max_num_sents
        self.max_sent_length = config.max_sent_length
        self.num_classes = config.num_classes - 1
        self.num_simlar_docs = config.num_simlar_docs
        self.num_concepts = config.num_concepts
        self.concept_dimesions = config.concept_dimesions
        self.vectors = vectors
        self.word_embedding_size = config.word_embedding_size
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.num_rnn_units = config.num_rnn_units
        self.attention_size = config.attention_size
        self.rnn_output_keep_prob = config.rnn_output_keep_prob
        self.text_field_names = config.text_field_names
        self.use_loss = config.use_loss
        self.threshold = config.threshold
        self.l2_reg_lambda = config.l2_reg_lambda
        #self.use_noise = config.use_noise
        self.input_doc = []
        self.doc_actual_num_sents = []
        self.doc_actual_sent_lengths = []
        for name in self.text_field_names:
            self.input_doc.append(tf.placeholder(tf.int32, [None, self.max_num_sents[name], self.max_sent_length[name]], name=name + '_input_doc'))
            self.doc_actual_num_sents.append(tf.placeholder(tf.int32, [None], name=name + '_doc_actual_num_sents'))
            self.doc_actual_sent_lengths.append(tf.placeholder(tf.int32, [None, self.max_num_sents[name]], name=name + '_doc_actual_sent_lengths'))

        self.input_cls = tf.placeholder(tf.float32, [None,self.num_classes], name='input_cls')
        self.cnn_dropout_keep_prob = tf.placeholder_with_default(1.0,[], name='cnn_dropout_keep_prob')
        self.attention_keep_prob = tf.placeholder_with_default(1.0,[], name='attention_keep_prob')
        self.rnn_output_keep_prob = tf.placeholder_with_default(1.0,[],name='rnn_output_keep_prob')
        self.attention_loss = tf.constant(0.0)
        self.l2_loss = tf.constant(0.0)
        self.fine_tune_word_embedding=config.fine_tune_word_embedding

        self.wv_initial = tf.constant(self.vectors, dtype=tf.float32)

    def add_embedding_layer(self,input):


        # Embedding layer
        with tf.variable_scope('embedding',reuse=tf.AUTO_REUSE):
            wordVectors = tf.get_variable('word_vectors', initializer=self.wv_initial,trainable=self.fine_tune_word_embedding)
            embedded_words = tf.nn.embedding_lookup(wordVectors, input)
        return embedded_words


    def add_bilstm_layer(self,input,actual_length,scope):
        #rnn context
        with tf.variable_scope('bilstm_' + scope, reuse=tf.AUTO_REUSE) :
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_rnn_units, forget_bias=1.0)
            print('fw cell',lstm_fw_cell)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=self.rnn_output_keep_prob)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_rnn_units, forget_bias=1.0)
            print('bw_cell',lstm_fw_cell)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=self.rnn_output_keep_prob)

            bilstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input,
                                                         actual_length, dtype=tf.float32)
        return bilstm_outputs
    def add_attention_layer(self,input, scope):
        with tf.name_scope('attention_layer_' + scope):
            attention_output, alphas = attention(input, self.attention_size, return_alphas=True)
            tf.summary.histogram('alphas', alphas)
            attention_output = tf.nn.dropout(attention_output, self.attention_keep_prob)
        return attention_output

    def sentence_features(self,input,actual_length):

        #sentence_input = tf.reshape(input,[-1,self.max_sent_length,self.word_embedding_size])
        sentence_bilstm_output = self.add_bilstm_layer(input,actual_length,'sent')
        sentence_features = self.add_attention_layer(sentence_bilstm_output,'sent')
        return sentence_features,2 * self.num_rnn_units
    def mask_sequence(self,input,max_length, input_actual_num,last_dimesion_size):

        mask = tf.to_float(tf.sequence_mask(input_actual_num, max_length))
        #print('input',input.get_shape())
        #print('mask',mask.get_shape())
        mask = tf.tile(tf.expand_dims(mask,-1),[1,1,last_dimesion_size])
        self.mask_shape = tf.shape(mask)
        masked = input * mask
        return masked
    def doc_features(self,input,actual_num_sents):

        doc_bilstm_output,_ = self.add_bilstm_layer(input,actual_num_sents,'doc')
        doc_features = self.add_attention_layer(doc_bilstm_output, 'doc')
        return doc_features
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

    def make_prediction(self,probabilities,threshold):
        pred = tf.where(tf.greater(probabilities,threshold),tf.ones_like(probabilities,dtype=tf.float32),tf.zeros_like(probabilities,dtype=tf.float32))
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
        doc_features = []
        for i, name in enumerate(self.text_field_names):

            input_embedded_words = self.add_embedding_layer(self.input_doc[i])

            sentence_input = tf.reshape(input_embedded_words, [-1, self.max_sent_length[name], self.word_embedding_size])
            actual_sent_lengths = tf.reshape(self.doc_actual_sent_lengths[i], [-1])
            sentence_features,sentence_num_features = self.sentence_features(sentence_input,actual_sent_lengths)



            if self.max_num_sents[name] > 1:
                sentence_features = tf.reshape(sentence_features, [-1, self.max_num_sents[name], sentence_num_features])
                doc_features.append(self.doc_features(sentence_features,self.doc_actual_num_sents[i]))
            else:
                doc_features.append(sentence_features)

        doc_features = tf.concat(doc_features, axis=-1)
        self.doc_prediction(doc_features, doc_features.get_shape()[-1],'doc_predict')


        self.add_loss()

