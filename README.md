# SongifyVox-Voice-to-Song
SongifyVox is an innovative voice conversion app powered by PyTorch, a cutting-edge deep learning framework. This revolutionary application allows users to transform their recorded voices by choosing their favorite AI artist into captivating songs with the help of AI algorithms.
# SongifyVox - Voice-to-Song Conversion App

SongifyVox is an exciting voice conversion application that uses the power of AI to transform recorded voices into captivating songs. This repository contains the source code for the SongifyVox app, along with the necessary Dockerfile and requirements.txt file for easy installation.

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
