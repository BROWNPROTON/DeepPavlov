"""
Copyright 2018 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from typing import List, Iterable, Union

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.tf_model import TFModel

log = get_logger(__name__)


class TensorflowBaseMatchingModel(TFModel):
    """
    Base class for ranking models that uses context-response matching schemes.

    Note:
        Tensorflow session variable already presents as self.sess attribute
        (derived from TFModel and initialized by Chainer)

    Args:
        batch_size (int): a number of samples in a batch

    """

    def __init__(self,
                 batch_size: int,
                 *args,
                 **kwargs):
        self.bs = batch_size
        super(TensorflowBaseMatchingModel, self).__init__(*args, **kwargs)

    def _append_sample_to_batch(self, sample,
                                batch_buffer_context,
                                batch_buffer_context_len,
                                batch_buffer_response,
                                batch_buffer_response_len):
        context_sentences = sample[:self.num_context_turns]
        response_sentences = sample[self.num_context_turns:]

        # Format model inputs:
        # 4 model inputs

        # 1. Token indices for context
        batch_buffer_context += [context_sentences for sent in response_sentences]
        # 2. Token indices for response
        batch_buffer_response += [response_sentence for response_sentence in response_sentences]
        # 3. Lens of context sentences
        lens = []
        for context in [context_sentences for sent in response_sentences]:
            context_sentences_lens = []
            for sent in context:
                context_sentences_lens.append(len(sent[sent != 0]))
            lens.append(context_sentences_lens)
        batch_buffer_context_len += lens
        # 4. Lens of context sentences
        lens = []
        for context in [response_sentence for response_sentence in response_sentences]:
            lens.append(len(context[context != 0]))
        batch_buffer_response_len += lens

    def _make_feed_dict(self, input_context,
                        input_context_len,
                        input_response,
                        input_response_len,
                        y=None
                        ):
        return {
            self.utterance_ph: np.array(input_context),
            self.all_utterance_len_ph: np.array(input_context_len),
            self.response_ph: np.array(input_response),
            self.response_len_ph: np.array(input_response_len),
            self.y_true: np.array(y)
        }

    def __call__(self, samples_generator: Iterable[List[np.ndarray]]) -> Union[np.ndarray, List[str]]:
        y_pred = []
        batch_buffer_context = []       # [batch_size, 10, 50]
        batch_buffer_context_len = []   # [batch_size, 10]
        batch_buffer_response = []      # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]
        j = 0
        while True:
            try:
                sample = next(samples_generator)
                j += 1
                n_responses = len(sample[self.num_context_turns:])
                self._append_sample_to_batch(sample,
                                             batch_buffer_context,
                                             batch_buffer_context_len,
                                             batch_buffer_response,
                                             batch_buffer_response_len)
                if len(batch_buffer_context) >= self.bs:
                    for i in range(len(batch_buffer_context) // self.bs):
                        fd = self._make_feed_dict(input_context=batch_buffer_context[i*self.bs:(i+1)*self.bs],
                                                  input_context_len=batch_buffer_context_len[i*self.bs:(i+1)*self.bs],
                                                  input_response=batch_buffer_response[i*self.bs:(i+1)*self.bs],
                                                  input_response_len=batch_buffer_response_len[i*self.bs:(i+1)*self.bs]
                                                  )
                        yp = self.sess.run(self.y_pred, feed_dict=fd)
                        y_pred += list(yp[:, 1])
                    lenb = len(batch_buffer_context) % self.bs
                    if lenb != 0:
                        batch_buffer_context = batch_buffer_context[-lenb:]
                        batch_buffer_context_len = batch_buffer_context_len[-lenb:]
                        batch_buffer_response = batch_buffer_response[-lenb:]
                        batch_buffer_response_len = batch_buffer_response_len[-lenb:]
                    else:
                        batch_buffer_context = []
                        batch_buffer_context_len = []
                        batch_buffer_response = []
                        batch_buffer_response_len = []
            except StopIteration:
                if j == 1:
                    return ["Error! It is not intended to use the model in the interact mode."]
                if len(batch_buffer_context) != 0:
                    fd = self._make_feed_dict(input_context=batch_buffer_context,
                                              input_context_len=batch_buffer_context_len,
                                              input_response=batch_buffer_response,
                                              input_response_len=batch_buffer_response_len
                                              )
                    yp = self.sess.run(self.y_pred, feed_dict=fd)
                    y_pred += list(yp[:, 1])
                break
        y_pred = np.asarray(y_pred)
        # reshape to [batch_size, 10] if needed
        y_pred = np.reshape(y_pred, (j, n_responses)) if n_responses > 1 else y_pred
        return y_pred

    # load() and save() are inherited from TFModel

    def train_on_batch(self, x: List[np.ndarray], y: List[int]) -> float:
        """
        This method is called by trainer to make one training step on one batch.

        :param x: generator that returns
                  list of ndarray - words of all sentences represented as integers,
                  with shape: (number_of_context_turns + 1, max_number_of_words_in_a_sentence)
        :param y: tuple of labels, with shape: (batch_size, )
        :return: value of loss function on batch
        """
        batch_buffer_context = []       # [batch_size, 10, 50]
        batch_buffer_context_len = []   # [batch_size, 10]
        batch_buffer_response = []      # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]
        j = 0
        while True:
            try:
                sample = next(x)
                j += 1         
                self._append_sample_to_batch(sample, 
                                             batch_buffer_context,
                                             batch_buffer_context_len, 
                                             batch_buffer_response, 
                                             batch_buffer_response_len)
                if len(batch_buffer_context) >= self.bs:
                    for i in range(len(batch_buffer_context) // self.bs):
                        fd = self._make_feed_dict(input_context=batch_buffer_context[i*self.bs:(i+1)*self.bs],
                                                  input_context_len=batch_buffer_context_len[i*self.bs:(i+1)*self.bs],
                                                  input_response=batch_buffer_response[i*self.bs:(i+1)*self.bs],
                                                  input_response_len=batch_buffer_response_len[i*self.bs:(i+1)*self.bs],
                                                  y=y
                                                  )
                        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=fd)
                    lenb = len(batch_buffer_context) % self.bs
                    if lenb != 0:
                        # keep the rest items in buffers if any
                        batch_buffer_context = batch_buffer_context[-lenb:]
                        batch_buffer_context_len = batch_buffer_context_len[-lenb:]
                        batch_buffer_response = batch_buffer_response[-lenb:]
                        batch_buffer_response_len = batch_buffer_response_len[-lenb:]
                    else:
                        batch_buffer_context = []
                        batch_buffer_context_len = []
                        batch_buffer_response = []
                        batch_buffer_response_len = []
            except StopIteration:
                if j == 1:
                    return ["Error! It is not intended to use the model in the interact mode."]
                if len(batch_buffer_context) != 0:
                    # feed the rest items
                    fd = self._make_feed_dict(input_context=batch_buffer_context,
                                              input_context_len=batch_buffer_context_len,
                                              input_response=batch_buffer_response,
                                              input_response_len=batch_buffer_response_len,
                                              y=y
                                              )
                    loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=fd)
                break
        return loss
