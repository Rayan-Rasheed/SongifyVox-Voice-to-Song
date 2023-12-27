# SongifyVox-Voice-to-Song
SongifyVox is an innovative voice conversion app powered by PyTorch, a cutting-edge deep learning framework. This revolutionary application allows users to transform their recorded voices by choosing their favorite AI artist into captivating songs with the help of AI algorithms.
# SongifyVox - Voice-to-Song Conversion App

SongifyVox is an exciting voice conversion application that uses the power of AI to transform recorded voices into captivating songs. This repository contains the source code for the SongifyVox app, along with the necessary Dockerfile and requirements.txt file for easy installation.

# How it works

1. **Voice Input**: The process starts with the user providing a voice input, which could be in the form of spoken words, singing, humming, or any vocal expression.
2. **Feature Extraction**: The input voice is analyzed, and relevant features are extracted, such as pitch, duration, and spectral characteristics. These features represent the unique characteristics of the user's voice.
3. **Data Preprocessing**: Before feeding the voice data to the conversion model, it undergoes preprocessing to normalize and standardize the features, ensuring consistency and optimal performance during conversion.
4. **Training Data**: A substantial amount of data is required to train the voice conversion model effectively. This data includes pairs of source voice (user's voice) and target voice (pre-recorded songs or a collection of different voices).
5. **Voice Conversion Model**: The heart of the process is the voice conversion model, typically based on deep learning techniques such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), or other neural network architectures.
6. **Learning Source-Target Mapping**: During training, the model learns the complex mapping between the source voice features (user's voice) and the corresponding target voice features (songs or other voices). This mapping allows the model to understand how to convert one voice into another convincingly.
7. **Conversion**: Once the model is trained, the actual conversion takes place. The user's voice features are fed into the model, and it applies the learned transformation to generate the target voice features.
8. **Synthesis**: The converted target voice features are synthesized to create the final output song. This synthesis involves reconstructing the vocal signals with the new target voice characteristics, resulting in a song that sounds like the user singing or speaking in the style of the target voice.
9. **Post-Processing**: The output may go through additional post-processing steps to refine the converted voice, enhance audio quality, or remove any artifacts introduced during conversion.
10. **Generated Song**: The end result is a fully transformed song or musical composition, which effectively reflects the user's voice while having the characteristics of the target voice or style.
**Technology Used**
Google Colab for AI artist training.
Visual Studio Code for making APIs in Python(3.10)
Pytorch and torch audio, torch creep for voice conversion and training
Vultr Nvidia GPU with Intel CPU's for the deployment of the application.
HTML, CSS, Javascript, and Bootstrap for Web Interface
Docker for containerizing the application
Following Python libraries are installation
torch
torchaudio
torchcrepe==0.0.15
torchgen==0.0.1
numpy==1.23.5


## Installation

To get started with SongifyVox, follow the steps below:

1. Clone this GitHub repository to your local machine:

```bash
clone https://github.com/your-username/SongifyVox.git
cd SongifyVox
```

1. Set up a virtual environment (optional but recommended):

```
python -m venv venv
"venv\Scripts\activate"
```

1. Install the required Python packages from the `requirements.txt` file:

   Windows with GPU to accelerate process

   ```bash
   python -m pip install -U pip setuptools wheel
   pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

   Other

   ```
   python -m pip install -U pip setuptools wheel
   pip install -U torch torchaudio 
   pip install -r requirements.txt
   ```

   Or simply

```
pip install -r requirements.txt
```

1. Build the Docker image using the provided Dockerfile:

```
docker build -t songifyvox .
```

## Usage

1. Ensure you have activated the virtual environment (if used) or are in the project directory.
2. Run the SongifyVox app using Docker:

```
docker run -p 8000:8000 songifyvox
```

1. Once the Docker container is up and running, visit `http://localhost:8000` in your web browser.
2. Follow the on-screen instructions to record your voice or upload pre-recorded clips.
3. Let SongifyVox work its magic and transform your voice into melodious songs!

## Customization

SongifyVox allows you to customize various aspects of the converted songs. Explore the `app` folder to modify musical styles, tempos, and instruments used in the generated compositions.

## Contributing

If you'd like to contribute to SongifyVox, we welcome your ideas and enhancements. Please fork this repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](https://chat.openai.com/LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.

## Acknowledgments

We would like to express our gratitude to the developers of the PyTorch library and other open-source packages that made this project possible.

Thank you for using SongifyVox! We hope you enjoy the magical experience of turning your voice into beautiful songs. For any issues or questions, please create an issue on this repository. Happy singing!
