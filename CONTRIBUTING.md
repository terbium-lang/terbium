# Contributing
We appreciate any contributions made to Terbium! 
Please ensure that your code meets the following requirements before contributing though:

- Have experience in both Rust and Terbium
- Run both `cargo check` and `cargo fmt` if you have modified Rust code
    - (`cargo fmt` will handle this) Make sure each line is under 120 characters long
    - We also advise you to run `cargo test` as well, but that will be tested by the GitHub workflow check.
      Doing such will probably save you an unnecessary commit if the check does fail.
    - For a similar reason, it is also advised that you run clippy with the following command as well:
      `cargo clippy --workspace --all-features -- -D clippy::all -D clippy::pedantic -D clippy::nursery`
      and make fixes accordingly.

## Tools needed
- Rust/Cargo (Nightly)
- LLVM 16.0+

## How do I install LLVM?

_If you are using macOS and have [Homebrew](https://brew.sh) installed, see [how to install LLVM using Homebrew](#using-homebrew)._

### From GitHub releases

1. Go to the [LLVM release page](https://github.com/llvm/llvm-project/releases) and find the latest release.

2. Pick the file that matches your system architecture.
    - For Unix/Linux platforms you want to pick the one that is prefixed with `clang+llvm-16.x.x`
    - For Windows pick [`LLVM-16.x.x-win32.exe`](https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/LLVM-16.0.4-win32.exe)
      or [`LLVM-16.x.x-win64.exe`](https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/LLVM-16.0.4-win64.exe)
      depending on your system architecture

3. Extract LLVM *if* you are on Unix/Linux platforms. 
   On Linux platforms you would run `tar -xf clang+llvm-16.x.x-the-file-that-I-downloaded.tar.xz`, 
   to make it easier to type later on, it is recommended to rename this file to `llvm-16.0.0`.
   **On Windows it is already an executable so no extraction is needed.**

4. You can place the extraced LLVM folder/executable wherever you want, if you can't decide, on Windows you can place it
   in `C:/Program Files/`, on Unix/Linux you can place it in `~/`

5. Then set environment variable `LLVM_SYS_160_PREFIX=/path/to/llvm/folder/` when you are compiling Terbium. 
   For example if the LLVM folder is at `/home/user/llvm-16.0.0` then I could build Terbium via:
   `LLVM_SYS_160_PREFIX=/home/user/llvm-16.0.0 cargo build --release`

6. Building Terbium should work now.

Happy Coding!

### Using Homebrew

*You must have [Homebrew](https://brew.sh) installed for this installation method. Homebrew is only available for macOS
and Linux; if you are using Windows consider installing from [GitHub releases](#from-github-releases).*

1. Install Homebrew if you haven't already (see https://brew.sh for instructions).

2. Run `brew install llvm@16` to install LLVM 16.

3. After installing, Homebrew may emit a *Caveats* section with instructions on how to use and link LLVM:
    ```text
    ==> Caveats
    To use the bundled libc++ please add the following LDFLAGS:
      LDFLAGS="-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++"
    
    llvm is keg-only, which means it was not symlinked into /opt/homebrew,
    because macOS already provides this software and installing another version in
    parallel can cause all kinds of trouble.
    
    If you need to have llvm first in your PATH, run:
      fish_add_path /opt/homebrew/opt/llvm/bin
    
    For compilers to find llvm you may need to set:
      set -gx LDFLAGS "-L/opt/homebrew/opt/llvm/lib"
      set -gx CPPFLAGS "-I/opt/homebrew/opt/llvm/include"
    ```

4. Run the commands in this section, especially the commands suggested after *For compilers to find llvm you may need to set...*:

    ```shell
    # These commands will vary. Refer to the Caveats section for the exact commands to run.
    $ set -gx LDFLAGS "-L/opt/homebrew/opt/llvm/lib"
    $ set -gx CPPFLAGS "-I/opt/homebrew/opt/llvm/include"
    ```

5. Then set environment variable `LLVM_SYS_160_PREFIX=/path/to/llvm/folder/` when you are compiling Terbium. 
   This is usually `/opt/homebrew/opt/llvm`.

6. Building Terbium should work now.
