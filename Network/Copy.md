To use `scp` to copy a directory but exclude a specific subdirectory in Debian 12, you can achieve this by combining `tar` with SSH. The `scp` command itself does not directly support excluding files or directories, but you can use the following approach as a workaround:

### Command:

```bash
tar --exclude='dir_father/dir_child1' -cf - dir_father | ssh user@remote_host 'cd /destination/path && tar -xf -'
```

### Explanation:
1. **`tar` on the local machine**:
   - `--exclude='dir_father/dir_child1'`: Skips the `dir_child1` directory during the `tar` process.
   - `-cf - dir_father`: Creates a tar archive of `dir_father` but writes it to standard output (`-`).
   
2. **`ssh`**:
   - Connects to the remote host.

3. **`tar` on the remote machine**:
   - `cd /destination/path`: Ensures the extracted files are placed in the correct destination.
   - `tar -xf -`: Reads the tar archive from standard input and extracts it.

Replace the following placeholders with actual values:
- `user`: Your username on the remote host.
- `remote_host`: The hostname or IP of the remote machine.
- `/destination/path`: The desired destination directory on the remote machine.

### Notes:
- Ensure you have the necessary permissions on both the local and remote directories.
- If your `dir_child1` contains sensitive data or large files, double-check the exclusion path to avoid accidental transfer.

Let me know if you need further clarification!