# Hugging Face Space MVP Deployment

This project includes a ready-to-push Hugging Face Space MVP in:

```text
hf_space_mvp/
```

The Space runs a Gradio UI for the current product schema:

- 3-class BSFS product grouping
- continuous Type 7 probability as a risk signal
- raw 7-class probabilities and top-3 output

## Local Verification

The Space app was smoke-tested locally by importing `app.py` and running inference on a clean-split test image.

```powershell
cd Model-Dev-main\hf_space_mvp
$env:PYTHONPATH='.'
@'
from pathlib import Path
from PIL import Image
import app
image_path = Path('..') / 'GuTelligence-StoMy-Clean-Split' / 'test' / 'Type 1' / 'test__type1_1_jpg.rf.a203c074ce6e75b9266fe03af3169bbe.jpg'
img = Image.open(image_path)
print(app.predict(img))
'@ | ..\.venv\Scripts\python.exe -
```

## Required Login

The local machine currently has the Hugging Face CLI package installed in the project virtual environment, but is not logged in.

Login command:

```powershell
cd Model-Dev-main\hf_space_mvp
..\.venv\Scripts\hf.exe auth login
```

Use a Hugging Face token with permission to create and write Spaces under the target organization.

## Create and Push

Recommended Space name:

```text
GuTelligence-Limited/bsfs-3class-type7-risk-mvp
```

Create the Space:

```powershell
..\.venv\Scripts\hf.exe repo create GuTelligence-Limited/bsfs-3class-type7-risk-mvp --type space --space-sdk gradio
```

Initialize and push the local Space repo:

```powershell
git init
git lfs install --local
git lfs track "*.pth"
git add .gitattributes .gitignore README.md requirements.txt model_registry.json app.py checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
git commit -m "Add BSFS 3-class Type7 risk MVP Space"
git remote add origin https://huggingface.co/spaces/GuTelligence-Limited/bsfs-3class-type7-risk-mvp
git branch -M main
git push -u origin main
```

## Large File Handling

The model checkpoint is stored in the Space repo through Git LFS:

```text
checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
```

Do not commit this checkpoint to the normal GitHub code repository. GitHub is used for source code and documentation; Hugging Face Space/Git LFS is used for the deployable model binary.
