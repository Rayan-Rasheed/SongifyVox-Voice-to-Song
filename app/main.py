from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
from rvcgui import *
from tempfile import NamedTemporaryFile
import base64

app = FastAPI()

# Configure CORS



app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://virtuvoxai.com","http://127.0.0.1:5500/",],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.post("/convert")
async def convert_voice(file: UploadFile = File(...), pitch: int = Form(...), model: str = Form(...), method: str = Form(...), creep_value: int = Form(...)):
    # Selecting models on model name
    selected_model(model)  # Successful

    # Getting temporary path of uploaded file
    file_extension = os.path.splitext(file.filename)[1]

    # Save the uploaded file to a temporary location with a specific name and extension
    with NamedTemporaryFile(suffix=file_extension, prefix="uploaded_", delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    audio_opt = on_button_click(temp_file_path, pitch, method, creep_value)

    # Output audio
    audio_data, target_sr = audio_opt

    # Save the output audio file within the container
    output_filename = "output.wav"
    output_file_path = os.path.join(os.getcwd(), output_filename)
    sf.write(output_file_path, audio_data, target_sr)

    # Read the contents of the output file as bytes
    with open(output_file_path, "rb") as output_file:
        output_data = output_file.read()
    output_base64 = base64.b64encode(output_data).decode("utf-8")
    # Clean up the temporary input file
    os.remove(temp_file_path)
    return {
        "audio_opt": output_base64,
        "message": "Audio processed successfully"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=800)
