in ubuntu, i use https git repository, each time i pull or push need to type in account and passward, how to avoid this? (cant use ssh)

To avoid typing your username and password each time you push or pull from a Git repository over HTTPS on Ubuntu, you can use Git's credential helper to cache your credentials. Here’s how you can set it up:

### 1. Use the Credential Cache Helper

The credential cache helper keeps your credentials in memory for a short period (default is 15 minutes).

Run the following command to enable it:

```sh
git config --global credential.helper cache
```

If you want to change the cache timeout, you can set it in seconds. For example, to cache credentials for 1 hour (3600 seconds):

```sh
git config --global credential.helper 'cache --timeout=3600'
```

### 2. Use the Credential Store Helper

The credential store helper saves your credentials in a plain text file on disk, which is more persistent but less secure than the cache method.

To enable the credential store helper:

```sh
git config --global credential.helper store
```

When you use `git pull` or `git push` for the first time after configuring this, Git will prompt you for your username and password, and then store them in a file at `~/.git-credentials`.

### 3. Use the libsecret Credential Helper

The `libsecret` credential helper integrates with GNOME Keyring to securely store your credentials.

First, install the required package:

```sh
sudo apt-get install libsecret-1-0 libsecret-1-dev
```

Then, you need to compile the `libsecret` credential helper. This is a one-time setup:

```sh
cd /usr/share/doc/git/contrib/credential/libsecret
sudo make
```

Finally, configure Git to use the `libsecret` helper:

```sh
git config --global credential.helper /usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret
```

### 4. Store Credentials for a Single Repository

If you want to store credentials for just one repository and not globally, navigate to your repository and run:

```sh
git config credential.helper store
```

### Steps to Verify

After configuring the credential helper, try pulling or pushing to your repository:

```sh
git pull
```

You should be prompted for your credentials the first time, and then they will be cached or stored based on your chosen method.

### Security Note

Storing credentials in plain text (using the store helper) can be insecure, as anyone with access to your file system can read them. Use the cache helper or `libsecret` for better security.