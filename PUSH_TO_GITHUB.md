# Push to GitHub

## Option A: push from this folder

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: hanacaraka nanoGPT"
git remote add origin <YOUR_PRIVATE_REPO_URL>
git push -u origin main
```

## Option B: push from the bundle file

```bash
git clone hanacaraka-nanogpt.bundle hanacaraka-nanogpt
cd hanacaraka-nanogpt
git remote add origin <YOUR_PRIVATE_REPO_URL>
git push -u origin main
```
