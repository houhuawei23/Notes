# Git

```bash
git remote set-url --add origin <git_url>

# You have divergent branches and need to specify how to reconcile them.
# You can do so by running one of the following commands sometime before
# your next pull:
 
git config pull.rebase false  # merge
git config pull.rebase true   # rebase
git config pull.ff only       # fast-forward only
```