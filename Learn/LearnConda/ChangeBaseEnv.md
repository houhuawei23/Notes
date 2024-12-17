# Change the Default Base Environment in Conda

## Problem:

on linux (debian12), how to change the default 'base' env in conda? current 'base' env use python12, i wnat use python11 as my default base env fo conda, how to do that?

## Solution:

To change the default `base` environment in Conda to use Python 3.11 instead of Python 3.12 on Debian 12, follow these steps:

---

### Step 1: Check Your Current `base` Environment

1. **Activate the `base` environment:**
   ```bash
   conda activate base
   ```
2. **Check the Python version:**
   ```bash
   python --version
   ```
   If it's showing Python 3.12, you'll need to update it.

---

### Step 2: Install Python 3.11 in the `base` Environment

1. **Update Conda to ensure you have the latest packages:**
   ```bash
   conda update conda
   conda update --all
   ```
2. **Replace the Python version in the `base` environment:**
   ```bash
   conda install python=3.11
   ```
   This will resolve dependencies and replace Python 3.12 with Python 3.11 in the `base` environment.

---

### Step 3: Verify the Change

1. **Confirm the Python version after installation:**

   ```bash
   python --version
   ```

   It should now show Python 3.11.

2. **Deactivate and reactivate `base` to ensure changes persist:**
   ```bash
   conda deactivate
   conda activate base
   ```

---

### Step 4: Set Conda to Always Use the `base` Environment on Start (Optional)

If you always want to activate the `base` environment when opening a terminal:

1. Edit your shell initialization file (e.g., `.bashrc` or `.zshrc`):
   ```bash
   nano ~/.bashrc
   ```
2. Add this line at the end:
   ```bash
   conda activate base
   ```
3. Save and reload your shell:
   ```bash
   source ~/.bashrc
   ```

---

Now, your Conda `base` environment will use Python 3.11 by default!
