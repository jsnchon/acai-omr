## Model Architecture

The final model, which I call `ViTOMR`, uses a ViT-B encoder and transformer decoder. The encoder outputs are fed through a linear projection to increase their dimension for use in cross attention. The decoder is 12 layers deep with a hidden dimension of 1024. The model has a total of 305,414,627 parameters.

I heavily customized these models for this task, primarly to address one of the most difficult issues in OMR: data. There aren't many OMR datasets and the ones that exist are extremely inconsistent, with images varying wildly in size. Instead of resizing all images to the same size, which would bring its own set of problems like distortion and detail loss, I wrote a lot of custom masking and padding logic to allow the transformers to deal with ragged batches, preventing the need to restrict all images to be the same size.

Additionally, I wrote my own custom attention layers and modules that allow for KV-caching, since PyTorch doesn't have native support for this. The code for this is in `acai_omr/models/kv_caching.py`.

The decoder outputs Linearized Musicxml tokens as created by the OLiMPiC dataset authors. These tokens can be delinearized using their scripts to get the final .musicxml file.

### ViTOMR Variants

Here's an overview of the variants of the base model I developed over my experiments with different training techniques. These models are implemented in `acai_omr/models/models.py`.

`TeacherForcedViTOMR` is used in standard teacher forcing training.

`ScheduledSamplingViTOMR` extends the teacher forced variant to support scheduled sampling, where the model's own predictions are fed to it during training, as opposed to an entirely ground-truth sequence.

`GRPOViTOMR` implements a lot of the logic for GRPO RL training. Note that the actual reward calculation and GRPO update steps aren't dealt with by the model itself.

### Scheduled Sampling Notes

The current version uses the scheduled sampling variant. To maintain differentiability in the sampling operation, allowing gradients to flow through the first forward pass, sampling is done using Gumbel-Softmax. More specifically, at each time step where sampling is being done, Gumbel-Softmax is applied to the logits outputted by the first forward pass, creating a probability distribution. Then, we set the input to the second forward pass at that time step to an expected embedding according to that distribution (unless we specify to use the straight-through estimation trick where discrete sampling is done on the forward pass and the Gumbel-Softmax function is used only to maintain differentiability in the backwards pass).

### KV Caching

`acai_omr/models/kv_caching.py` contains the implementation for my KV-cached module variants. These have a separate cached pathway where the flow is as follows:
1. The keys and values from the encoded image latent is stored in separate cross-attention caches, one for each layer. This means we only need to run the encoder once instead of redundantly calculating key and value tensors from the image latent for each new token
2. Self-attention caches, again one for each layer, store past key and value tensors from previous text tokens
