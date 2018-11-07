# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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

import numpy as np
from typing import List, Iterable

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from deeppavlov.models.ranking.matching_models.tf_base_matching_model import TensorflowBaseMatchingModel
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad_truncate

log = get_logger(__name__)


@register('matching_predictor')
class MatchingPredictor(Component):
    """The class for ranking of the response given N context turns
    using the trained SMN or DAM neural network in the ``interact`` mode.

    Args:
        num_context_turns (int): A number N of ``context`` turns in data samples.
        max_sequence_length (int): A maximum length of text sequences in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        *args, **kwargs: Other parameters.
    """

    def __init__(self,
                 model: TensorflowBaseMatchingModel,
                 num_context_turns: int = 10,
                 max_sequence_length: int = 50,
                 *args, **kwargs) -> None:

        super(MatchingPredictor, self).__init__()

        self.num_context_turns = num_context_turns
        self.max_sequence_length = max_sequence_length
        self.model = model

    def __call__(self, batch: Iterable[List[np.ndarray]]) -> List[str]:
        """
        Overrides __call__ method.

        Args:
            batch (Iterable): A batch of one sample, preprocessed, but not padded to ``num_context_turns`` sentences

        Return:
             list of verdict messages
        """
        sample = next(batch)
        try:
            next(batch)
            log.error("It is not intended to use the `%s` with the batch size greater then 1." % self.__class__)
        except StopIteration:
            pass

        batch_buffer_context = []       # [batch_size, 10, 50]
        batch_buffer_context_len = []   # [batch_size, 10]
        batch_buffer_response = []      # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]

        preproc_sample = []
        for s in sample:
            preproc_sample.append(s.tolist())

        sent_list = zero_pad_truncate(preproc_sample, self.max_sequence_length, pad='post', trunc='post')

        # reshape the sample that contains multiple responses to a batch contains n rows like ([context, response1],...)
        n_responses = len(sent_list[self.num_context_turns:])
        self.model._append_sample_to_batch(sent_list,
                                           batch_buffer_context,
                                           batch_buffer_context_len,
                                           batch_buffer_response,
                                           batch_buffer_response_len)
        if len(batch_buffer_context) >= self.bs:
            for i in range(len(batch_buffer_context) // self.bs):
                fd = self.model._make_feed_dict(
                    input_context=batch_buffer_context[i * self.bs:(i + 1) * self.bs],
                    input_context_len=batch_buffer_context_len[i * self.bs:(i + 1) * self.bs],
                    input_response=batch_buffer_response[i * self.bs:(i + 1) * self.bs],
                    input_response_len=batch_buffer_response_len[i * self.bs:(i + 1) * self.bs]
                    )
                yp = self.model.sess.run(self.model.y_pred, feed_dict=fd)
                y_pred = list(yp[:, 1])
        lenb = len(batch_buffer_context) % self.bs
        if lenb != 0:
            fd = self._make_feed_dict(input_context=batch_buffer_context[-lenb:],
                                      input_context_len=batch_buffer_context_len[-lenb:],
                                      input_response=batch_buffer_response[-lenb:],
                                      input_response_len=batch_buffer_response_len[-lenb:]
                                      )
            yp = self.sess.run(self.y_pred, feed_dict=fd)
            y_pred += list(yp[:, 1])
        y_pred = np.asarray(y_pred)
        # reshape to [batch_size, 10] if needed
        y_pred = np.reshape(y_pred, (1, n_responses)) if n_responses > 1 else y_pred

        # return ["The probability that the response is proper continuation of the dialog is {:.3f}".format(y_pred[0])]
        # return ["{:.5f}".format(y_pred[0])]
        vector = y_pred[0]
        str_vector = []
        for i in vector:
            str_vector.append("{:.5f}".format(i))
        return str_vector

    def reset(self) -> None:
        pass

    def process_event(self) -> None:
        pass
