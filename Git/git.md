# Git

- [Commands](#commands)
- [Q & A](#q--a)

<br>

- [git-scm](https://git-scm.com/)

## Commands

- [`git branch`](#git-branch)
- [`git remote`](#git-remote)
- [`git push`](#git-push)
- [`git merge`](#git-merge)
- [`git config`](#git-config)
- [`git rebase`](#git-rebase)
- [`git reset`](#git-reset)

### `git branch`

```bash
# List all branches (local and remote; the current branch is highlighted by *):
git branch --all
# Create new branch based on the current commit:
git branch branch_name
# Create new branch based on a specific commit:
git branch branch_name commit_hash
# Rename a branch (must not have it checked out to do this):
git branch -m|--move old_branch_name new_branch_name
# Delete a local branch (must not have it checked out to do this):
git branch -d|--delete branch_name
# Delete a remote branch:
git push remote_name --delete remote_branch_name
```

### `git remote`

```bash
# chekout remotes
git remote -v
# Show information about a remote:
git remote show remote_name
# Add a remote:
git remote add remote_name remote_url
# Change the URL of a remote (use --add to keep the existing URL):
git remote set-url remote_name new_url
# Remove a remote:
git remote remove remote_name
# Rename a remote:
git remote rename old_name new_name

# add additional url to the remote branch named `origin`
# after that, the `origin` branch will has multi url
# if run `git push origin main`, git will push to both of them
git remote set-url --add origin <git_url>
```

### `get push`

Push commits to a remote repository.

```bash
# Send local changes in the current branch to its default remote counterpart:
git push
# Send changes from a specific local branch to its remote counterpart:
git push remote_name local_branch
# Send changes from a specific local branch to its remote counterpart,
# and set the remote one as the default push/pull target of the local one:
git push -u remote_name local_branch # --set-upstream
```

### [`git merge`](https://git-scm.com/docs/git-merge)

Merge branches

```bash
# Merge a branch into your current branch:
git merge branch_name

# Edit the merge message:
git merge --edit branch_name

# Merge a branch and create a merge commit:
git merge --no-ff branch_name

# Abort a merge in case of conflicts:
git merge --abort

# Merge using a specific strategy:
git merge --strategy strategy --strategy-option strategy_option branch_name

# !! merge to only create a commit on top of current commit
git merge --squash branch_name
```

#### `--ff, --no-ff, --ff-only`

Specifies how a merge is handled when the merged-in history is already a descendant of the current history. --ff is the
default unless merging an annotated (and possibly signed) tag that is not stored in its natural place in the refs/tags/
hierarchy, in which case --no-ff is assumed.

- With --ff, when possible resolve the merge as a fast-forward (only update the branch pointer to match the merged branch; do
not create a merge commit). When not possible (when the merged-in history is not a descendant of the current history), create
a merge commit.

- With --no-ff, create a merge commit in all cases, even when the merge could instead be resolved as a fast-forward.

- With --ff-only, resolve the merge as a fast-forward when possible. When not possible, refuse to merge and exit with a
non-zero status.

#### `--squash, --no-squash`

Produce the working tree and index state as if a real merge happened (except for the merge information), but do not actually make a commit, move the HEAD, or record $GIT_DIR/MERGE_HEAD (to cause the next git commit command to create a merge commit).

This allows you to create a single commit on top of the current branch whose effect is the same as merging another branch (or more in case of an octopus).

### `git config`

```bash
# You have divergent branches and need to specify how to reconcile them.
# You can do so by running one of the following commands sometime before
# your next pull:

git config pull.rebase false  # merge
git config pull.rebase true   # rebase
git config pull.ff only       # fast-forward only
```

### [`git rebase`](https://git-scm.com/docs/git-rebase)

Reapply commits from one branch on top of another branch.

Commonly used to "move" an entire branch to another base, creating copies of the commits in the new location.

```bash
# Rebase the current branch on top of another specified branch:
git rebase new_base_branch

# Start an interactive rebase, which allows the commits to be reordered, omitted, combined or modified:
git rebase -i|--interactive target_base_branch_or_commit_hash
```

### [`git reset`](https://git-scm.com/docs/git-reset)

- Undo commits or unstage changes, by resetting the current Git HEAD to the specified state.
- If a path is passed, it works as "unstage"; if a commit hash or branch is passed, it works as "uncommit".

```bash
# Unstage everything:
git reset

# Unstage specific file(s):
git reset path/to/file1 path/to/file2 ...

# Interactively unstage portions of a file:
git reset --patch path/to/file

# Undo the last commit, keeping its changes (and any further uncommitted changes) in the filesystem:
git reset HEAD~

# Undo the last two commits, adding their changes to the index, i.e. staged for commit:
git reset --soft HEAD~2

# Discard any uncommitted changes, staged or not (for only unstaged changes, use git checkout):
git reset --hard

# !! Reset the repository to a given commit, discarding committed, staged and uncommitted changes since then:
git reset --hard commit
```

## Q & A

### How to set default upstream branch for `git push` or `git pull`?

1 checkout which branch your local branch is tracking:

```bash
git branch -vv
```

2 set upstream branch for local branch

```bash
git branch --set-upstream-to=origin/remote-branch-name local-branch-name
```

3 unset the upstream branch

```bash
git branch --unset-upstream local-branch-name
```

### squash multiple commits into one to create a cleaner pull request

- upstream rep/branch 'abc/main',
- my rep/branch 'forked-abc/main' and 'forked-abc/dev'

how to do:

- we can develop on 'forked-abc/dev' with multiple commits
- then checkout 'forked-abc/main' and merge 'forked-abc/dev' to it
- then rebase 'forked-abc/main' on top of 'abc/main'
  - squash all the commits on 'forked-abc/main' into one commit
- finally i can create a pull request to 'abc/main' with only one conbined commit

how:

```bash
# Add the original repository as a remote
git remote add upstream https://github.com/abc/main.git

# Fetch latest changes
git fetch upstream

# Rebase your branch onto the latest upstream branch
git checkout your-branch # main
# git merge dev
git rebase upstream/main

# Squash commits
git rebase -i upstream/main
# Mark commits as `pick` or `squash`, then save

# Force-push the squashed commit
git push --force-with-lease origin your-branch

# Create a pull request
# Go to your forked repository and create a pull request to the original repository
```

### How to avoid enter password every time when pushing to a remote repository (https)?

A: Use Git's credential helper to cache your credentials.

> Security Note!!! \
> Storing credentials in plain text (using the store helper) can be insecure, as anyone with access to your file system can read them. Use the cache helper or `libsecret` for better security.

#### 1. Use the Credential Cache Helper

The credential cache helper keeps your credentials in memory for a short period (default is 15 minutes).

```bash
# enable credential cache helper
git config --global credential.helper cache
# change cache timeout (in seconds)
git config --global credential.helper 'cache --timeout=3600'
```

#### 2. Use the Credential Store Helper

The credential store helper saves your credentials in a plain text file on disk, which is more persistent but less secure than the cache method.

```sh
# enable the credential store helper
git config --global credential.helper store
```

When you use `git pull` or `git push` for the first time after configuring this, Git will prompt you for your username and password, and then store them in a file at `~/.git-credentials`.

#### 3. Use the `libsecret` Credential Helper

The `libsecret` credential helper integrates with GNOME Keyring to securely store your credentials.

```sh
# First, install the required package:
sudo apt-get install libsecret-1-0 libsecret-1-dev
# Then, you need to compile the `libsecret` credential helper.
# This is a one-time setup:
cd /usr/share/doc/git/contrib/credential/libsecret
sudo make
# Finally, configure Git to use the `libsecret` helper:
git config --global credential.helper /usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret
```

#### 4. Store Credentials for a Single Repository

If you want to store credentials for just one repository and not globally, navigate to your repository and run:

```sh
git config credential.helper store
```
