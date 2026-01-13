# Image Caption Generator with CNN-LSTM

This project implements an image captioning model using a Convolutional Neural Network (CNN) and a Long Short-Term Memory (LSTM) network. The architecture is inspired by the "Show and Tell" model, with modifications to improve performance. The model is trained on the Flickr8K dataset.

## Methodology

The model follows an encoder-decoder structure. A CNN acts as the encoder to extract features from the image, and an LSTM network acts as the decoder to generate a descriptive caption.

### 1. Image Feature Extraction (Encoder)

-   **Model**: A pre-trained **DenseNet201** model, originally trained on the ImageNet dataset, is used for feature extraction.
-   **Process**: Each image is resized to 224x224 pixels and fed into the DenseNet201 model.
-   **Feature Vector**: The output from the final Global Average Pooling layer of the CNN is used as the image feature vector. This results in a 1920-dimensional vector that provides a rich representation of the image's contents. These features are pre-computed and stored before training the decoder.

### 2. Text Preprocessing

The captions from the Flickr8K dataset are preprocessed before being used for training:
-   Converted to lowercase.
-   Removal of special characters and numbers.
-   Tokenization of sentences into words.
-   Addition of `startseq` and `endseq` tokens to each caption to signify the beginning and end of a sequence.

### 3. Decoder Architecture (LSTM)

The decoder is an LSTM network responsible for generating the caption word-by-word.

-   **Inputs**: The decoder takes two inputs at each time step:
    1.  The 1920-dimensional image feature vector from the encoder.
    2.  The sequence of words generated so far, starting with `startseq`.
-   **Architecture Details**:
    1.  The image feature vector is passed through a `Dense` layer to reduce its dimensionality to 256.
    2.  The input text sequence is passed through an `Embedding` layer to create a 256-dimensional vector for each word.
    3.  The condensed image vector is concatenated with the word embedding sequence. This combined sequence is fed into an `LSTM` layer with 256 units.
    4.  **Key Modification**: In a departure from the original "Show and Tell" architecture, the output of the LSTM is added to the condensed image vector. This residual-style connection reinforces the image context throughout the generation process.
    5.  The combined vector is then passed through `Dense` layers and a final `softmax` activation function to predict the next word in the sequence from the entire vocabulary.

### 4. Training

-   **Data Generation**: A custom Keras `Sequence` data generator is used to feed data in batches, which is memory-efficient for large datasets.
-   **Optimizer**: The model is compiled with the `Adam` optimizer.
-   **Loss Function**: `categorical_crossentropy` is used as the loss function, suitable for multi-class classification (predicting the next word).
-   **Callbacks**: Several callbacks are used to manage the training process:
    -   `ModelCheckpoint`: Saves the model with the best validation loss.
    -   `EarlyStopping`: Halts training if the validation loss does not improve for 5 consecutive epochs.
    -   `ReduceLROnPlateau`: Reduces the learning rate if the validation loss plateaus.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/subhashchapa/image_caption_generator.git
    cd image_caption_generator
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas tensorflow matplotlib seaborn
    ```

3.  **Download the Dataset:**
    Download the Flickr8K dataset. You will need the `Images` folder and the `captions.txt` file. Make sure the paths in the `CNN-LSTM project.ipynb` notebook point to the correct locations of these files.

4.  **Run the Jupyter Notebook:**
    Open and execute the `CNN-LSTM project.ipynb` notebook in a Jupyter environment.

## Future Work

-   Train the model on a larger dataset (e.g., Flickr30k or MS COCO) to improve generalization.
-   Implement an **Attention mechanism** to allow the model to focus on more relevant parts of the image while generating captions.
-   Evaluate the model using standard metrics like **BLEU score**.
