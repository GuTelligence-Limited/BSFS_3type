# Hugging Face Space MVP Deployment

This project includes a ready-to-push Hugging Face Space MVP in:

```text
hf_space_mvp/
```

The Space runs a Gradio UI for the current product schema:

- 3-class BSFS product grouping
- continuous Type 7 probability as a risk signal
- raw 7-class probabilities and top-3 output
- model architecture summary through `torchinfo`

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

Current MVP Space:

```text
https://huggingface.co/spaces/perram27/bsfs-3class-type7-risk-mvp
```

The original target organization Space was:

```text
GuTelligence-Limited/bsfs-3class-type7-risk-mvp
```

Creating under `GuTelligence-Limited` failed because the logged-in Hugging Face user did not have organization write permission. The MVP Space was therefore created under the logged-in user namespace `perram27`.

Create the Space under a namespace with write permission:

```powershell
..\.venv\Scripts\hf.exe repos create perram27/bsfs-3class-type7-risk-mvp --type space --space-sdk gradio --exist-ok
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

If Git HTTPS push is blocked or times out, upload small files through the Hub API:

```powershell
..\.venv\Scripts\hf.exe upload perram27/bsfs-3class-type7-risk-mvp app.py app.py --repo-type space --commit-message "Add Gradio app"
..\.venv\Scripts\hf.exe upload perram27/bsfs-3class-type7-risk-mvp requirements.txt requirements.txt --repo-type space --commit-message "Add requirements"
..\.venv\Scripts\hf.exe upload perram27/bsfs-3class-type7-risk-mvp model_registry.json model_registry.json --repo-type space --commit-message "Add model registry"
```

After architecture-tab updates, upload these files:

```powershell
..\.venv\Scripts\hf.exe upload perram27/bsfs-3class-type7-risk-mvp app.py app.py --repo-type space --commit-message "Add model architecture tab"
..\.venv\Scripts\hf.exe upload perram27/bsfs-3class-type7-risk-mvp requirements.txt requirements.txt --repo-type space --commit-message "Add torchinfo dependency"
```

## Large File Handling

The model checkpoint is stored in the Space repo through Git LFS:

```text
checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
```

Do not commit this checkpoint to the normal GitHub code repository. GitHub is used for source code and documentation; Hugging Face Space/Git LFS is used for the deployable model binary.

Current checkpoint upload status:

- Small Space files are uploaded.
- The `.pth` checkpoint upload timed out repeatedly from the current network.
- The Space will not run inference until the checkpoint exists at:

```text
checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
```

Retry checkpoint upload from a stable network:

```powershell
cd C:\gutelligence\cv_model\Model-Dev-main\hf_space_mvp
..\.venv\Scripts\hf.exe upload perram27/bsfs-3class-type7-risk-mvp checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth --repo-type space --commit-message "Add ConvNeXt Tiny checkpoint"
```
