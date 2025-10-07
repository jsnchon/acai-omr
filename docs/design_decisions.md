## Design Decisions

Essentially every implementation decision made within this project was motivated by two big challenges in OMR: 

1. The nature of the task. Music notation is extremely varied, complex, and high-precision.    
   - For a given symbol, the smallest deviation in its position or the presence of another tiny symbol beside it can completely alter its semantic meaning and the meanings of other symbols around it 
   - For more complex polyphonic sheets, the model must do more than simple visual recognition of symbols like in Optical Character Recognition. It must also identify different voices and separate notes between them which requires at least some semantic understanding of musi c
2. Data issues.
    - There's a general lack of datasets
    - Datasets that do exist are usually limited in size and/or difficult to work with for various reasons, eg image sizes vary wildly within/between them, lack of image diversity, etc. 
    - There is nothing that comes close to being a generally-accepted foundation model

I outline my thought process behind a lot of the decisions made over the development of this project below.
    
#### Why transformers?

Having a transformer decoder was basically a given since this is a sequence-to-sequence problem where the output is text. That being said, the encoder didn't necessarily have to be a ViT, and a CNN would have some advantages. I ended up using a ViT though for a few reasons:

- For my own education I wanted to work with ViTs which I had never really done seriously before
- There are cool self-supervised techniques available for transformer training I wanted to play with/learn about
- ViTs make some intuitive sense for the task: musical information in sheets is contextually dependent and influenced by long-distance relationships, eg the time signature all the way at the start of a system will influence the amount of notes in each measure indefinitely

#### Why modify the transformers?

I had to choose between forcibly resizing every image to a fixed dimension or heavily modifying the transformers to natively support batches containing images of different sizes. Resizing would have been much, much easier, but I ultimately decided that minimally resizing each image was crucial in order to preserve aspect ratio and image detail, both of which are very important for this task.

With easier datasets to work with, it would be easy to set a fixed dimension for all images that only required minimal resizing for individual examples. But the aforementioned data issues meant this would be very difficult in my case -- some images ranged from ~150 x 300 pixels to ~1000 x 1700, and aspect ratios varied wildly from ~0.05 to ~25. 

Plus it was a fun challenge and really improved my skills with PyTorch tensor ops.

#### Why pretrain the encoder?

Training the encoder and decoder from scratch simultaneously on the smallish labelled dataset was simply not feasible, so there was a clear need for an encoder that could encode musical images well before the supervised training stage.

As mentioned before, there are no real OMR foundation models. Additionally, pretrained ViTs are trained on datasets with huge domain shift compared to my task. So to ensure I had an encoder that could actually produce useful latent representations for sheet music, I decided to pretrain my own.

#### Why output Linearized Musicxml (LMX)?

[Mayer et al. 2024](github.com/ufal/olimpic-icdar24/tree/master?tab=readme-ov-file) discuss the advantages of their encoding system in detail. There were a few that were especially important for this project:
- Musicxml files are extremely verbose which would make decoding slow and expensive
- I would have to figure out some sort of tokenization scheme for Musicxml, whereas LMX's vocabulary is small and well-defined
- LMX makes it easier to deal with the complexities of pianoform music, especially with representing multiple overlapping voices 

#### Why scheduled sampling?

Intuitively, scheduled sampling seems especially important in this difficult, precise visual task where it's easy to make mistakes during autoregressive inference. Scheduled sampling provides a way for the model to learn to deal with its own past mistakes.

And ultimately, scheduled sampling was the training method that achieved the best performance and generalization to somewhat out of distribution images, eg photos printed sheets.
