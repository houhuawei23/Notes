curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
info: downloading installer

Welcome to Rust!

This will download and install the official compiler for the Rust
programming language, and its package manager, Cargo.

Rustup metadata and toolchains will be installed into the Rustup
home directory, located at:

  /home/hhw/.rustup

This can be modified with the RUSTUP_HOME environment variable.

The Cargo home directory is located at:

  /home/hhw/.cargo

This can be modified with the CARGO_HOME environment variable.

The cargo, rustc, rustup and other commands will be added to
Cargo's bin directory, located at:

  /home/hhw/.cargo/bin

This path will then be added to your PATH environment variable by
modifying the profile files located at:

  /home/hhw/.profile
  /home/hhw/.bashrc
  /home/hhw/.zshenv

You can uninstall at any time with rustup self uninstall and
these changes will be reverted.

Current installation options:


   default host triple: x86_64-unknown-linux-gnu
     default toolchain: stable (default)
               profile: default
  modify PATH variable: yes

1) Proceed with standard installation (default - just press enter)
2) Customize installation
3) Cancel installation

info: profile set to 'default'
info: default host triple is x86_64-unknown-linux-gnu
info: syncing channel updates for 'stable-x86_64-unknown-linux-gnu'
info: latest update on 2024-08-08, rust version 1.80.1 (3f5fd8dd4 2024-08-06)
info: downloading component 'cargo'
  8.2 MiB /   8.2 MiB (100 %)   2.0 MiB/s in  8s ETA:  0s
info: downloading component 'clippy'
  2.4 MiB /   2.4 MiB (100 %)   1.8 MiB/s in  1s ETA:  0s
info: downloading component 'rust-docs'
 15.8 MiB /  15.8 MiB (100 %)   3.0 MiB/s in  5s ETA:  0s
info: downloading component 'rust-std'
 26.7 MiB /  26.7 MiB (100 %)   2.2 MiB/s in 12s ETA:  0s
info: downloading component 'rustc'
 65.0 MiB /  65.0 MiB (100 %)   3.5 MiB/s in 22s ETA:  0s
info: downloading component 'rustfmt'
info: installing component 'cargo'
info: installing component 'clippy'
info: installing component 'rust-docs'
 15.8 MiB /  15.8 MiB (100 %)  12.9 MiB/s in  1s ETA:  0s
info: installing component 'rust-std'
 26.7 MiB /  26.7 MiB (100 %)  18.3 MiB/s in  1s ETA:  0s
info: installing component 'rustc'
 65.0 MiB /  65.0 MiB (100 %)  19.4 MiB/s in  3s ETA:  0s
info: installing component 'rustfmt'
info: default toolchain set to 'stable-x86_64-unknown-linux-gnu'

  stable-x86_64-unknown-linux-gnu installed - rustc 1.80.1 (3f5fd8dd4 2024-08-06)


Rust is installed now. Great!

To get started you may need to restart your current shell.
This would reload your PATH environment variable to include
Cargo's bin directory ($HOME/.cargo/bin).

To configure your current shell, you need to source
the corresponding env file under $HOME/.cargo.

This is usually done by running one of the following (note the leading DOT):
. "$HOME/.cargo/env"            # For sh/bash/zsh/ash/dash/pdksh
source "$HOME/.cargo/env.fish"  # For fish