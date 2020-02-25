# CHANGELOG

All notable changes to this project will be documented in this file.

### v0.1.0
* Added `ulangel.data.text_processor` to clean the text.
* Added `freeze_all`, `unfreeze_all`, `freeze_upto` in `ulangel.utils.learner` to be able to freeze or unfreeze a certain number of layers in a neural network.
* Added textplus input mode. In v0.0.1 there was just one input mode: text only mode (taking only verbal features), since v0.1.0 there is a new input mode: text plus mode (not only verbal features but also nonverbal features can be inputs of the neural network).
* Replaced `pad_collate` in `ulangel.data.data_packer` by `pad_collate_textonly` to do the padding for text only inputs and `pad_collate_textplus` for text plus inputs.
* Replaced `SentenceEncoder` in `ulangel.rnn.nn_block` by `TextOnlySentenceEncoder` for text only inputs and `TextPlusSentenceEncoder` for text plus inputs.
* Replaced `PoolingLinearClassifier` in `ulangel.rnn.nn_block` by `TextOnlyPoolingLinearClassifier` for text only inputs and `TextPlusPoolingLinearClassifier` for text plus inputs.
* Replaced `CudaCallback` in `ulangel.utils.callbacks` by `TextOnlyCudaCallback` for text only inputs and `TextPlusCudaCallback` for text plus inputs.
