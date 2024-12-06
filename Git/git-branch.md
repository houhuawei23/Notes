To set a default upstream branch for `git push` and `git pull`, you need to configure the branch to track a specific remote branch. Here’s how to do it:

---

### **Set Default Upstream Branch**

1. **Using `git push` Command**
   The easiest way to set an upstream branch is to use the `--set-upstream` option with `git push`:

   ```bash
   git push --set-upstream origin <branch-name>
   ```

   This command does two things:
   - Pushes your branch to the remote repository.
   - Configures the branch to track the corresponding branch on the remote.

2. **Using `git branch` Command**
   You can manually set the upstream branch for the current branch:

   ```bash
   git branch --set-upstream-to=origin/<remote-branch> <local-branch>
   ```

   Example:

   ```bash
   git branch --set-upstream-to=origin/main main
   ```

3. **While Creating a New Branch**
   If you create a new branch and want it to track a remote branch from the start:

   ```bash
   git checkout -b <branch-name> --track origin/<remote-branch>
   ```

   Example:

   ```bash
   git checkout -b feature/new-feature --track origin/feature/new-feature
   ```

---

### **Verify Upstream Branch**

To confirm the upstream branch configuration:

```bash
git branch -vv
```

- The output will display the upstream branch associated with each local branch.

---

### **Remove or Change Upstream Branch**

1. **To Remove an Upstream Branch**
   If you no longer want a branch to track a remote branch:

   ```bash
   git branch --unset-upstream
   ```

2. **To Change the Upstream Branch**
   Use the `--set-upstream-to` option to point to a different remote branch:

   ```bash
   git branch --set-upstream-to=origin/new-branch
   ```

---

### **How it Works for `git push` and `git pull`**

- Once the upstream branch is set, `git push` and `git pull` will automatically use the configured upstream branch.
- Example workflow:
  1. Set upstream:

     ```bash
     git push --set-upstream origin main
     ```

  2. Push and pull without specifying branches:

     ```bash
     git push
     git pull
     ```

Let me know if you need further assistance!
