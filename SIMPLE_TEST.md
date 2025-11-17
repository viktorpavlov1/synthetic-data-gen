# Simple One-Command Test

## Just run this:

### On Windows:
```bash
python install_and_test.bat
```

Or if that doesn't work, run these commands one by one:
```bash
pip install --upgrade tokenizers transformers
pip install -r requirements.txt
pip install -e .
python test.py
```

### On Mac/Linux:
```bash
bash install_and_test.sh
```

Or manually:
```bash
pip install --upgrade tokenizers transformers
pip install -r requirements.txt
pip install -e .
python test.py
```

## What it does:

The `test.py` script will:
1. ✅ Check all dependencies are installed
2. 🖼️ Generate 2 synthetic skin lesion images
3. 🔍 Classify them using ResNet50
4. 📊 Show you the results

**That's it!** Just one command and you'll see if everything works.

## If you get errors:

1. **Tokenizers error**: Run `pip install --upgrade tokenizers transformers` first
2. **Module not found**: Run `pip install -e .` to install the package
3. **GPU issues**: The script uses small images (256x256) and should work on CPU too (just slower)

